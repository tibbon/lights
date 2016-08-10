from bottle import route, run, template
import redis
import cPickle
import json
import struct
import os

# Set browser to 36 x 60

def redis_conn():
    host = os.environ.get('REDISHOST', '192.168.1.101')
    return redis.Redis(host=host)

def prep_client(client):
    client.delete('q1')
    client.delete('q2')
    client.delete('q3')
    client.delete('q4')

client = redis_conn()
prep_client(client)

# pixel_width=116

@route('/')
def index():
    return open('main.html').read()

@route('/clear')
def data():
    prep_client(client)
    return 'ok'

@route('/jquery.js')
def jquery():
    return open('jquery.js').read()

@route('/data')
def data():
    _, q1_frame = client.blpop('q1')
    _, q2_frame = client.blpop('q2')
    _, q3_frame = client.blpop('q3')
    _, q4_frame = client.blpop('q4')

    q1_frame = cPickle.loads(q1_frame)
    q2_frame = cPickle.loads(q2_frame)
    q3_frame = cPickle.loads(q3_frame)
    q4_frame = cPickle.loads(q4_frame)

    frame = q1_frame + q2_frame + q3_frame + q4_frame

    if frame == None:
        return json.dumps(None)

    # frame = cPickle.loads(result)
    frame = ''.join(frame)
    frame = map(lambda x: struct.unpack('!B', x)[0], frame)

    dump = []

    for i in range(0, len(frame), 3):
        dump.append([frame[i], frame[i+1], frame[i+2]])

    return json.dumps(dump)

run(host='localhost', port=8080, debug=True)
