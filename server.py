from http.server import SimpleHTTPRequestHandler, HTTPServer

PORT = 8000
Handler = SimpleHTTPRequestHandler

def run(server_class=HTTPServer, handler_class=Handler):
    server_address = ('', PORT)
    httpd = server_class(server_address, handler_class)
    print(f'Serving on port {PORT}...')
    httpd.serve_forever()

if __name__ == "__main__":
    run()
