from flask import Flask
from flask_restful import Resource, Api
from flask import request
app = Flask(__name__)
api = Api(app)


class HelloWorld(Resource):
    def get(self):
        print("hello")

        return {'name': 'akul '}
    

@app.route('/image',methods = ['POST', 'GET'])
def image():
    print("post")
    print(request.data) 
    f = open("RequestData.txt",'w')
    f.write((request.data).decode("utf-8"))
    f.close()
    return "{'name': 'post'}"


api.add_resource(HelloWorld, '/')

if __name__ == '__main__':
    app.run(debug=True)
