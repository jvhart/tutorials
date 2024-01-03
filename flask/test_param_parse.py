from flask import Flask
from flask_restful import Resource, Api, reqparse

app = Flask(__name__)
api = Api(app)


class BarAPI(Resource):
    def get(self):

        parser = reqparse.RequestParser()
        parser.add_argument('key1', type=str)
        parser.add_argument('key2', type=str)

        return parser.parse_args()


api.add_resource(BarAPI, '/bar', endpoint='bar')

if __name__ == '__main__':
    app.run(debug=True)
