import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import api


class ApiRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(api.app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_health_and_local_home_catalog_work_without_omdb_key(self):
        health = self.client.get("/health")
        home = self.client.get("/home", params={"category": "popular", "limit": 3})

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json(), {"status": "ok"})
        self.assertEqual(home.status_code, 200)
        self.assertEqual(len(home.json()), 3)
        self.assertTrue(all(movie["imdb_id"] is None for movie in home.json()))

    def test_search_reports_missing_omdb_key(self):
        with patch.object(api, "_get_omdb_api_key", return_value=None):
            response = self.client.get("/omdb/search", params={"query": "Matrix"})

        self.assertEqual(response.status_code, 503)
        self.assertIn("OMDB_API_KEY", response.json()["detail"])

    def test_search_returns_omdb_results(self):
        payload = {
            "Response": "True",
            "Search": [
                {
                    "imdbID": "tt0133093",
                    "Title": "The Matrix",
                    "Year": "1999",
                    "Poster": "https://example.test/matrix.jpg",
                }
            ],
        }
        with patch.object(
            api, "omdb_search_movies", new=AsyncMock(return_value=payload)
        ):
            response = self.client.get("/omdb/search", params={"query": "Matrix"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["Search"][0]["imdbID"], "tt0133093")


class OmdbMappingTests(unittest.IsolatedAsyncioTestCase):
    async def test_movie_details_are_mapped_to_app_schema(self):
        payload = {
            "imdbID": "tt0133093",
            "Title": "The Matrix",
            "Year": "1999",
            "Released": "31 Mar 1999",
            "Genre": "Action, Sci-Fi",
            "Plot": "A computer hacker learns the truth.",
            "Poster": "https://example.test/matrix.jpg",
        }
        with patch.object(api, "omdb_get", new=AsyncMock(return_value=payload)):
            details = await api.omdb_movie_details("tt0133093")

        self.assertEqual(details.imdb_id, "tt0133093")
        self.assertEqual(details.title, "The Matrix")
        self.assertEqual([genre["name"] for genre in details.genres], ["Action", "Sci-Fi"])

    async def test_invalid_key_error_is_actionable_without_exposing_key(self):
        error = api._omdb_error({"Error": "Invalid API key!"})

        self.assertEqual(error.status_code, 401)
        self.assertIn("OMDB_API_KEY", error.detail)
        self.assertNotIn("Invalid API key!", error.detail)


if __name__ == "__main__":
    unittest.main()
