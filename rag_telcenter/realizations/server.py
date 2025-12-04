# from . import RAG

# help_text = """RAG Telcenter is running.

# It uses the JSON-RPC 2.0 protocol over HTTP.

# Requests should be sent as POST requests with a JSON body
# to endpoint /api/v1/json-rpc.

# Request payload:

# ```json
# {
#   "jsonrpc": "2.0",
#   "method": "subtract",
#   "params": [10, 5],
#   "id": 1
# }
# ```

# Response payload (on success):

# ```json
# {
#   "jsonrpc": "2.0",
#   "result": 5,
#   "id": 1
# }
# ```

# Methods:

# - update_dataframe: Update the RAG vectorstore system with a new dataframe.
# - query_vectordb: Query the vector database.
# - query_reasoning: Query the reasoning engine.
# """

# def serve():
#     from flask import Flask

#     app = Flask(__name__)

#     @app.route("/")
#     def index():
#         return 
