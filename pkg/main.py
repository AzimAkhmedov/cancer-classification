from http.server import HTTPServer
from controller import Controller

PORT = 8000


httpd = HTTPServer(("", PORT), Controller)

print(f"Serving at port {PORT}")
httpd.serve_forever()