import os
import shutil
import glob

def process_tianchi_dir(path):
    files = glob.glob(os.path.join(path, '*.zip'))
    for file in files:
        os.unlink(file)

    data = os.path.join(path, 'data')
    remove_all_files(data)

    tmp = os.path.join(path, 'tmp_data')
    remove_all_files(tmp)

    merge = os.path.join(path, 'merge_data')
    remove_all_files(merge)

    video = os.path.join(path, 'video_data')
    remove_all_files(video)



def remove_all_files(data_):
    for f in os.listdir(data_):
        if os.path.isfile(os.path.join(data_, f)):
            os.unlink(os.path.join(data_, f))
        elif os.path.islink(os.path.join(data_, f)):
            os.unlink(os.path.join(data_, f))
        elif os.path.isdir(os.path.join(data_, f)):
            shutil.rmtree(os.path.join(data_, f))

if __name__ == '__main__':
    path = r'/workspace/solo/code/user_data/'
    process_tianchi_dir(path)