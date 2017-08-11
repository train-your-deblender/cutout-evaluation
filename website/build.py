#!/usr/bin/env python
from __future__ import print_function
import sys
import time
import logging
import threading
import subprocess
from watchdog.observers import Observer
from watchdog.events import LoggingEventHandler, FileSystemEventHandler
import os.path
import posixpath
import urllib

try:
    import BaseHTTPServer
except ImportError:
    import http.server as BaseHTTPServer

try:
    import SimpleHTTPServer
except ImportError:
    import http.server as SimpleHTTPServer

try:
    from urllib import unquote
except ImportError:
    from urllib.parse import unquote

BIND_TO = ('127.0.0.1', 8000)

THEME_DIR = os.path.abspath(os.path.dirname(__file__))
SCSS_FILE_PATH = os.path.join(THEME_DIR, 'static', 'styles.scss')
CSS_OUTPUT_PATH = os.path.join(THEME_DIR, 'static', 'styles.css')

class RequestHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
    document_root = os.path.join(THEME_DIR)
    
    def translate_path(self, path):
        # Borrowed from the Python standard library implementation of SimpleHTTPRequestHandler
        
        # abandon query parameters
        path = path.split('?', 1)[0]
        path = path.split('#', 1)[0]
        # Don't forget explicit trailing slash when normalizing. Issue17324
        trailing_slash = path.rstrip().endswith('/')
        path = posixpath.normpath(unquote(path))
        words = path.split('/')
        words = filter(None, words)
        path = self.document_root
        for word in words:
            drive, word = os.path.splitdrive(word)
            head, word = os.path.split(word)
            if word in (os.curdir, os.pardir):
                continue
            path = os.path.join(path, word)
        if trailing_slash:
            path += '/'
        return path

def compile_scss():
    proc = subprocess.Popen(
        ['sassc', '-m', SCSS_FILE_PATH, CSS_OUTPUT_PATH],
        stdout=subprocess.PIPE,
        bufsize=1,
        cwd=THEME_DIR
    )
    buf = []
    for line in iter(proc.stdout.readline, ''):
        buf.append(line)
        # logging.info(line, end='')  # uncomment this to enable output
    proc.stdout.close()
    returncode = proc.wait()
    if returncode != 0:
        logging.info('*'*72)
        logging.info('sassc exited with returncode != 0 (was {})'.format(returncode))
        logging.info('*'*72)
        logging.info('\n'.join(buf))
        logging.info('*'*72)
    logging.info('Build completed.')

class SCSSRebuildHandler(FileSystemEventHandler):
    def on_any_event(self, event):
        if '.scss' in event.src_path:
            compile_scss()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    path = os.path.join(THEME_DIR, 'static')
    # Serve static files
    server = BaseHTTPServer.HTTPServer(BIND_TO, RequestHandler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    # Watch for filesystem events and rebuild docs as necessary
    event_handler = LoggingEventHandler()
    rebuild_handler = SCSSRebuildHandler()

    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.schedule(rebuild_handler, path, recursive=True)
    observer.start()
    
    # Do an initial build, and open a browser window
    compile_scss()
    url = "http://{}:{}/demo.html".format(*BIND_TO)
    import webbrowser
    webbrowser.open(url)
    logging.info("*"*72)
    logging.info("\tNow serving the preview at {}".format(url))
    logging.info("*"*72)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
