import base64
import hmac
import time

import requests


with open("token.txt", "r") as f:  # 打开文件
    data = f.read()  # 读取文件
    print(data)

'''连接tlink平台数据的认证信息'''
userid = "200000152"
thinkAppId = "36b4943878cb4f8eab2c6473f0169917"
Authorization = str("Bearer ") + str(data)
print(Authorization)
headers = {"tlinkAppId": thinkAppId, "Authorization": Authorization, "Content-type": "application/json",
           "Cache-Control": "no-cache"}


def getinfo(json, url):
    r = requests.post(url, data=json, headers=headers)
    res = r.text
    return res


def generate_token(key, expire=3600):
    r'''
        @Args:
            key: str (用户给定的key，需要用户保存以便之后验证token,每次产生token时的key 都可以是同一个key)
            expire: int(最大有效时间，单位为s)
        @Return:
            state: str
    '''
    ts_str = str(time.time() + expire)
    ts_byte = ts_str.encode("utf-8")
    sha1_tshexstr = hmac.new(key.encode("utf-8"), ts_byte, 'sha1').hexdigest()
    token = ts_str + ':' + sha1_tshexstr
    b64_token = base64.urlsafe_b64encode(token.encode("utf-8"))
    return b64_token.decode("utf-8")


def certify_token(key, token):
    r'''
        @Args:
            key: str
            token: str
        @Returns:
            boolean
    '''
    token_str = base64.urlsafe_b64decode(token).decode('utf-8')
    token_list = token_str.split(':')
    if len(token_list) != 2:
        return False
    ts_str = token_list[0]
    if float(ts_str) < time.time():
        # token expired
        return False
    known_sha1_tsstr = token_list[1]
    sha1 = hmac.new(key.encode("utf-8"), ts_str.encode('utf-8'), 'sha1')
    calc_sha1_tsstr = sha1.hexdigest()
    if calc_sha1_tsstr != known_sha1_tsstr:
        # token certification failed
        return False
    # token certification success
    return True
