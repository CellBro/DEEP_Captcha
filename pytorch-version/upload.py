import requests
f = open(r"22920212204263_best.txt", "rb")
files = {'file':f}
r = requests.post(url = "http://121.37.1.35:5001/detectfile",files = files)
print(r.text)
