import unittest
from server.app import create_app

class TestFlaskApp(unittest.TestCase):
    def setUp(self):
        app_server = create_app()
        self.client = app_server.test_client()


    def test_no_start_node(self):
        response = self.client.post("/server", json={})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"Error": "No entry node is specified"})

    # @unittest.skip("This test is skipped intentionally")
    def test_no_destinaton_node(self):
        response = self.client.post("/server", json={"e1": "1"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"Error": "No desination node in the graph is specified"})

    def test_no_graph(self):
        response = self.client.post("/server", json={"e1": "1", "h": "5"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"Error": "No graph is specified"})

    def test_bad_parsing(self):
        response = self.client.post("/server", json={"e1": "1","h": "5","graph": "digraph graphname{\n1->2\n2->3\n2-asdfsdf>5\n5->2\n3->5"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"Error": "Error in parsing dot string"})


    def test_good_parsing(self):
        response = self.client.post("/server", json={"e1": "1","h": "5","graph": "digraph graphname{\n1->2\n2->3\n2->5\n5->2\n3->5}"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(),  {"come_before_node": ["1", "2"]})


if __name__ == '__main__':
    unittest.main()
