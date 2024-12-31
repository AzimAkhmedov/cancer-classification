from http.server import SimpleHTTPRequestHandler


class Controller(SimpleHTTPRequestHandler):
     def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h1>Welcome to My Server!</h1></body></html>")
        else:
            self.send_error(404, "File Not Found")

