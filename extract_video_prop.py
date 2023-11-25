import subprocess

# print('This is standard output', file=sys.stdout)
def getVideoDetail(video_name):
    cmd = ["ffprobe -show_streams {}".format(video_name)]

    p = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)

    out, _ = p.communicate()
    decoded_val = out.decode('UTF-8')
    key_value_pair = {'DISPOSITION': {}, 'TAG': {}}
    tokens = decoded_val.split("\n")
    for t in tokens:
        if len(t) != 0:
            result = t.split("=")
            if len(result) == 2:
                key = result[0]
                value = result[1]
                if 'DISPOSITION' in key or 'TAG' in key:
                    new_key = result[0].split(':')

                    key_value_pair[new_key[0]].update({new_key[1]: value})
                else:
                    key_value_pair.update({key:value})
    return key_value_pair

if __name__== "__main__":
    getVideoDetail('/app/datasets/videos/886/Sujan_Adhikari.mp4')
    # print(getVideoDetail('/app/datasets/videos/886/Sujan_Adhikari.mp4'))
