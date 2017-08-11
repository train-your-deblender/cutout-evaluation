# The random bytes used to secure cookies. By default this is a new random
# number every time you start the Notebook. Set it to a value in a config file
# to enable logins to persist across server sessions.
#
# Note: Cookie secrets should be kept private, do not share config files with
# cookie_secret stored in plaintext (you can read the value from a file).

c.NotebookApp.cookie_secret = b'95903671659027190548421207374194900598390713828425540093085327436163704928371'

# The file where the cookie secret is stored.
# c.NotebookApp.cookie_secret_file = ''

# Hashed password to use for web authentication.
#
# To generate, type in a python/IPython shell:
#
#   from notebook.auth import passwd; passwd()
#
# The string should be of the form type:salt:hashed-password.
c.NotebookApp.password = 'sha1:9b2503c71801:bd43d50c74a3441e77ca0ce76fd429753406104d'

# Whether to open in a browser after starting. The specific browser used is
# platform dependent and determined by the python standard library `webbrowser`
# module, unless it is overridden using the --browser (NotebookApp.browser)
# configuration option.
c.NotebookApp.open_browser = False

# The port the notebook server will listen on.
c.NotebookApp.port = 9900

# Bind to all interfaces:
c.NotebookApp.ip = '0.0.0.0'
