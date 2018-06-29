#!/usr/bin/env python
 
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs
from sklearn.externals import joblib
import numpy as np

model = 0
template = 0

# HTTPRequestHandler class
class testHTTPServer_RequestHandler(BaseHTTPRequestHandler):
 
  # GET
  def do_GET(self):
        qs = parse_qs( urlparse(self.path).query )
        print("request", self.path, qs)
        # Send response status code
        self.send_response(200)
 
        # Send headers
        self.send_header('Content-type','text/html')
        self.end_headers()
 
        if 'RM' in qs and 'LSTAT' in qs and 'PTRATIO' in qs:
            RM = float(qs['RM'][0])
            LSTAT = float(qs['LSTAT'][0])
            PTRATIO = float(qs['PTRATIO'][0])
            #PRICE = (RM * 86565.23 + LSTAT * (-10849.34) + PTRATIO * (-19492.11) + 415464.39)
            PRICE = model.predict(np.array([RM, LSTAT, PTRATIO]).reshape(1,-1))
            print('Calculating', PRICE)
            content = template.replace("SERVER PREDICTION", "SERVER PREDICTION: {:.2f}".format(PRICE[0]))
        else:
            content = template.replace("SERVER PREDICTION", "")
        # Write content as utf-8 data
        self.wfile.write(bytes(content, "utf8"))
        return
 
def run():
    global model
    global template

    print('Loading template ...')
    template = open('predict-price.html', 'r').read()

    print('Loading decision tree ...')
    model = joblib.load('dtr_model.pkl')

    print('starting server...')

    # Server settings
    # Choose port 8080, for port 80, which is normally used for a http server, you need root access
    server_address = ('127.0.0.1', 8081)
    httpd = HTTPServer(server_address, testHTTPServer_RequestHandler)
    print('running server...')
    httpd.serve_forever()

 
run()