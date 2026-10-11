import asyncio
from contextlib import asynccontextmanager
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd
from dotenv import dotenv_values, load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

OMDB_BASE = "https://www.omdbapi.com/"

@asynccontextmanager
async def lifespan(_: FastAPI):
   load_pickles()
   yield


app = FastAPI(title="Movie Recommender API", version="4.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")

df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None
TITLE_TO_IDX: Optional[Dict[str, int]] = None
OMDB_CACHE: Dict[Tuple[Tuple[str, str], ...], Dict[str, Any]] = {}


class MovieCard(BaseModel):
    imdb_id: Optional[str] = None
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None


class MovieDetails(BaseModel):
    imdb_id: str
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = Field(default_factory=list)


class TFIDFRecItem(BaseModel):
    title: str
    score: float
    movie: Optional[MovieCard] = None


class SearchBundleResponse(BaseModel):
    query: str
    movie_details: MovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[MovieCard]


def _norm_title(title: str) -> str:
    return str(title).strip().lower()


def _get_omdb_api_key() -> Optional[str]:
    return os.getenv("OMDB_API_KEY") or dotenv_values(
        os.path.join(BASE_DIR, ".env")
    ).get("OMDB_API_KEY")


def _omdb_error(data: Dict[str, Any]) -> HTTPException:
    message = str(data.get("Error") or "OMDb returned an unexpected response.")
    lowered = message.lower()
    if "invalid api key" in lowered:
        return HTTPException(
            status_code=401,
            detail="OMDb rejected the API key. Check OMDB_API_KEY in the project .env file.",
        )
    if "request limit reached" in lowered:
        return HTTPException(status_code=429, detail=message)
    return HTTPException(status_code=502, detail=f"OMDb error: {message}")


async def omdb_get(params: Dict[str, Any]) -> Dict[str, Any]:
    api_key = _get_omdb_api_key()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="OMDB_API_KEY is not set. Add your activated key to the project .env file and restart the API.",
        )

    cache_key = tuple(sorted((key, str(value)) for key, value in params.items()))
    if cache_key in OMDB_CACHE:
        return OMDB_CACHE[cache_key]

    request_params = dict(params)
    request_params["apikey"] = api_key
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(15, connect=5)) as client:
            response = await client.get(OMDB_BASE, params=request_params)
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not connect to OMDb ({type(exc).__name__}). Check your internet connection.",
        ) from exc

    if response.status_code in (401, 403):
        raise HTTPException(
            status_code=401,
            detail="OMDb rejected the API key. Check OMDB_API_KEY in the project .env file.",
        )
    if response.status_code == 429:
        raise HTTPException(status_code=429, detail="OMDb request limit reached.")
    if response.status_code != 200:
        raise HTTPException(
            status_code=502, detail=f"OMDb returned HTTP {response.status_code}."
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="OMDb returned invalid JSON.") from exc

    if str(data.get("Response", "")).lower() == "false":
        message = str(data.get("Error", ""))
        if "movie not found" not in message.lower():
            raise _omdb_error(data)

    OMDB_CACHE[cache_key] = data
    return data


async def omdb_search_movies(query: str, page: int = 1) -> Dict[str, Any]:
    return await omdb_get({"s": query, "type": "movie", "page": page})


def movie_card_from_omdb(data: Dict[str, Any]) -> MovieCard:
    rating = data.get("imdbRating")
    try:
        vote_average = float(rating) if rating and rating != "N/A" else None
    except (TypeError, ValueError):
        vote_average = None

    poster = data.get("Poster")
    return MovieCard(
        imdb_id=data.get("imdbID"),
        title=data.get("Title") or "",
        poster_url=poster if poster and poster != "N/A" else None,
        release_date=data.get("Released") or data.get("Year"),
        vote_average=vote_average,
    )


async def omdb_movie_details(imdb_id: str) -> MovieDetails:
    data = await omdb_get({"i": imdb_id, "plot": "full"})
    if not data.get("imdbID"):
        raise HTTPException(status_code=404, detail="Movie not found in OMDb.")

    genres = [
        {"name": genre.strip()}
        for genre in str(data.get("Genre") or "").split(",")
        if genre.strip() and genre.strip() != "N/A"
    ]
    poster = data.get("Poster")
    return MovieDetails(
        imdb_id=data["imdbID"],
        title=data.get("Title") or "",
        overview=data.get("Plot") if data.get("Plot") != "N/A" else None,
        release_date=data.get("Released") or data.get("Year"),
        poster_url=poster if poster and poster != "N/A" else None,
        genres=genres,
    )


async def omdb_card_by_title(title: str) -> Optional[MovieCard]:
    data = await omdb_search_movies(title)
    results = data.get("Search") or []
    if not results:
        return None

    match = next(
        (item for item in results if _norm_title(item.get("Title", "")) == _norm_title(title)),
        results[0],
    )
    return movie_card_from_omdb(match)


def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    if not hasattr(indices, "items"):
        raise RuntimeError(
            "indices.pkl must be a dict or pandas Series-like object with .items()."
        )
    return {_norm_title(title): int(index) for title, index in indices.items()}


def get_local_idx_by_title(title: str) -> int:
    if TITLE_TO_IDX is None:
        raise HTTPException(status_code=500, detail="TF-IDF index map not initialized.")
    index = TITLE_TO_IDX.get(_norm_title(title))
    if index is None:
        raise HTTPException(
            status_code=404, detail=f"Title not found in local dataset: '{title}'."
        )
    return index


def tfidf_recommend_titles(
    query_title: str, top_n: int = 10
) -> List[Tuple[str, float]]:
    if df is None or tfidf_matrix is None:
        raise HTTPException(status_code=500, detail="TF-IDF resources not loaded.")

    index = get_local_idx_by_title(query_title)
    scores = (tfidf_matrix @ tfidf_matrix[index].T).toarray().ravel()
    order = np.argsort(-scores)

    recommendations: List[Tuple[str, float]] = []
    for candidate_index in order:
        if int(candidate_index) == index:
            continue
        title = str(df.iloc[int(candidate_index)]["title"])
        recommendations.append((title, float(scores[int(candidate_index)])))
        if len(recommendations) >= top_n:
            break
    return recommendations


def local_genre_titles(
    genres: List[dict], excluded_title: str, limit: int
) -> List[str]:
    if df is None or not genres or "genres" not in df.columns:
        return []

    genre_names = {
        token
        for genre in genres
        for token in str(genre["name"]).lower().replace("-", " ").split()
    }
    rows = df.copy()
    rows["_popularity"] = pd.to_numeric(rows["popularity"], errors="coerce")
    matches = rows[
        rows["genres"].fillna("").map(
            lambda value: bool(
                genre_names.intersection(
                    {
                        token
                        for token in str(value).lower().replace("-", " ").split()
                    }
                )
            )
        )
    ].sort_values("_popularity", ascending=False, na_position="last")

    titles: List[str] = []
    seen = {_norm_title(excluded_title)}
    for title in matches["title"]:
        normalized = _norm_title(title)
        if normalized and normalized not in seen:
            seen.add(normalized)
            titles.append(str(title))
        if len(titles) >= limit:
            break
    return titles


async def cards_for_titles(titles: List[str]) -> List[MovieCard]:
    semaphore = asyncio.Semaphore(5)

    async def fetch_card(title: str) -> Optional[MovieCard]:
        async with semaphore:
            return await omdb_card_by_title(title)

    results = await asyncio.gather(*(fetch_card(title) for title in titles))
    return [card for card in results if card is not None]


def load_pickles():
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX

    with open(DF_PATH, "rb") as file:
        df = pickle.load(file)
    with open(INDICES_PATH, "rb") as file:
        indices_obj = pickle.load(file)
    with open(TFIDF_MATRIX_PATH, "rb") as file:
        tfidf_matrix = pickle.load(file)
    with open(TFIDF_PATH, "rb") as file:
        tfidf_obj = pickle.load(file)

    TITLE_TO_IDX = build_title_to_idx_map(indices_obj)
    if df is None or "title" not in df.columns:
        raise RuntimeError("df.pkl must contain a DataFrame with a 'title' column.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/home", response_model=List[MovieCard])
async def home(
    category: str = Query("popular"),
    limit: int = Query(24, ge=1, le=50),
):
    if df is None:
        raise HTTPException(status_code=500, detail="Movie dataset is not loaded.")
    if category not in {"popular", "top_rated"}:
        raise HTTPException(status_code=400, detail="Choose popular or top_rated.")

    column = "popularity" if category == "popular" else "vote_average"
    movies = df[["title", column]].copy()
    movies[column] = pd.to_numeric(movies[column], errors="coerce")
    movies = movies.dropna(subset=["title", column]).sort_values(
        column, ascending=False
    )

    cards: List[MovieCard] = []
    seen = set()
    for row in movies.itertuples(index=False, name=None):
        title = str(row[0]).strip()
        if not title or _norm_title(title) in seen:
            continue
        seen.add(_norm_title(title))
        cards.append(
            MovieCard(
                title=title,
                vote_average=float(row[1]) if category == "top_rated" else None,
            )
        )
        if len(cards) >= limit:
            break
    return cards


@app.get("/omdb/search")
@app.get("/tmdb/search", include_in_schema=False)
async def movie_search(
    query: str = Query(..., min_length=1),
    page: int = Query(1, ge=1, le=100),
):
    return await omdb_search_movies(query.strip(), page=page)


@app.get("/movie/id/{imdb_id}", response_model=MovieDetails)
async def movie_details_route(imdb_id: str):
    if not imdb_id.startswith("tt"):
        raise HTTPException(status_code=400, detail="Expected an IMDb ID such as tt0133093.")
    return await omdb_movie_details(imdb_id)


@app.get("/recommend/genre", response_model=List[MovieCard])
async def recommend_genre(
    imdb_id: str = Query(..., min_length=3),
    limit: int = Query(18, ge=1, le=50),
):
    details = await omdb_movie_details(imdb_id)
    titles = local_genre_titles(details.genres, details.title, limit)
    return await cards_for_titles(titles)


@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=50),
):
    recommendations = tfidf_recommend_titles(title, top_n=top_n)
    return [{"title": title, "score": score} for title, score in recommendations]


@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
    query: str = Query(..., min_length=1),
    imdb_id: Optional[str] = Query(None, min_length=3),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30),
):
    if imdb_id:
        details = await omdb_movie_details(imdb_id)
    else:
        search_data = await omdb_search_movies(query)
        results = search_data.get("Search") or []
        if not results:
            raise HTTPException(
                status_code=404, detail=f"No OMDb movie found for query: {query}"
            )
        details = await omdb_movie_details(results[0]["imdbID"])

    try:
        recommendations = tfidf_recommend_titles(details.title, top_n=tfidf_top_n)
    except HTTPException as exc:
        if exc.status_code != 404:
            raise
        recommendations = []

    tfidf_items: List[TFIDFRecItem] = []
    if recommendations:
        cards = await cards_for_titles([title for title, _ in recommendations])
        cards_by_title = {_norm_title(card.title): card for card in cards}
        tfidf_items = [
            TFIDFRecItem(
                title=title,
                score=score,
                movie=cards_by_title.get(_norm_title(title)),
            )
            for title, score in recommendations
        ]

    genre_titles = local_genre_titles(details.genres, details.title, genre_limit)
    genre_cards = await cards_for_titles(genre_titles)
    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_cards,
    )
