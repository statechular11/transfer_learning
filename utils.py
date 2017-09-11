import os
import sys
import shutil
import errno
import imghdr
import requests
from io import BytesIO
from keras import backend as K


# define constants
IMAGE_EXTENSIONS = [
    'rgb', 'gif', 'pbm', 'pgm', 'ppm', 'tiff',
    'rast', 'xbm', 'jpg', 'jpeg', 'bmp', 'png'
]


def create_dir(dirpath):
    """
    Create a directory if it does not exist
    """
    try:
        os.makedirs(os.path.expanduser(dirpath))
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
        elif not os.path.isdir(dirpath):
            print(('Warning: `' + dirpath + '` already exists '
                   'and is not a directory!'))


def remove_file(filepath):
    """
    Remove a file if exists
    """
    try:
        os.remove(os.path.expanduser(filepath))
    except OSError as exception:
        if exception.errno != errno.ENOENT:
            raise


def ceildiv(dividend, divisor):
    """ceiling-division for two integers
    """
    return -(-dividend // divisor)


def file_of_extensions(filepath, ext_list):
    """
    Check the extension of a file is one of given list of extensions
    """
    _, ext = os.path.splitext(filepath)
    return (ext[1:]).lower() in ext_list


def files_under_dir(dirpath):
    """Return a list of all files in a given directory and its subdirectories
    """
    abs_path = os.path.abspath(os.path.expanduser(dirpath))
    files = [os.path.join(root, file)
             for root, dirs, files in os.walk(abs_path)
             for file in files]
    return sorted(files)


def files_under_subdirs(dirpath, subdirs=None):
    """
    Return a list of all files in a given directory and specified subdirs
    """
    abs_path = os.path.abspath(os.path.expanduser(dirpath))
    if not subdirs:
        subdirs = os.listdir(abs_path)
    subdirs = [os.path.basename(subdir) for subdir in subdirs]
    files = [file for subdir in subdirs
             for file in files_under_dir(os.path.join(abs_path, subdir))]
    return sorted(files)


def images_under_dir(dirpath,
                     examine_by='extension',
                     ext_list=IMAGE_EXTENSIONS):
    """
    Return a list of image files in a given dir and its subdirs

    Parameters
    ----------
    dirpath: string or unicode
        path to the directory
    examine_by: string (default='extension')
        method of examining image file, either 'content' or 'extension'
        If 'content', `imghdr.what()` is used
        If 'extension', the file extension is compared against a list of common
        image extensions
    ext_list: list, optional (used only when examine_by='extension')
        If examine_by='extension', the file extension is compared against
        ext_list
    """
    files = files_under_dir(dirpath)
    assert examine_by in ['content', 'extension']
    if examine_by == 'content':
        images = [file for file in files if imghdr.what(file)]
    else:
        images = [file for file in files if file_of_extensions(file, ext_list)]
    return images


def images_under_subdirs(dirpath,
                         subdirs=None,
                         examine_by='extension',
                         ext_list=IMAGE_EXTENSIONS):
    """
    Return a list of image files in a given dir and specified subdirs
    """
    files = files_under_subdirs(dirpath, subdirs)
    assert examine_by in ['content', 'extension']
    if examine_by == 'content':
        images = [file for file in files if imghdr.what(file)]
    else:
        images = [file for file in files if file_of_extensions(file, ext_list)]
    return images


def get_keras_backend_name():
    try:
        return K.backend()
    except AttributeError:
        return K._BACKEND


def set_image_format():
    try:
        if K.backend() == 'theano':
            K.set_image_data_format('channels_first')
        else:
            K.set_image_data_format('channels_last')
    except AttributeError:
        if K._BACKEND == 'theano':
            K.set_image_dim_ordering('th')
        else:
            K.set_image_dim_ordering('tf')


def url2file(url):
    """
    Read a URL and return a file(-like) object, which can be further provided
    to `keras.preprocessing.image.load_img()`
    """
    try:
        response = requests.get(url)
        file_obj = BytesIO(response.content)
        return file_obj
    except requests.exceptions.RequestException as e:
        print('Can read {}'.format(url))
        raise e
        sys.exit(1)


def move_files_between_dirs(src, dst):
    """
    Move all the files under origin directory to destination directory
    """
    create_dir(dst)
    for filename in os.listdir(src):
        shutil.move(os.path.join(src, filename), os.path.join(dst, filename))


def copy_files_between_dirs(src, dst):
    """
    Copy all the files under origin directory to destination directory
    """
    if not os.path.exists(dst):
        shutil.copytree(src, dst)
        return
    for filename in os.listdir(src):
        try:
            shutil.copytree(os.path.join(src, filename),
                            os.path.join(dst, filename))
        except OSError as exc:
            if exc.errno == errno.ENOTDIR:
                shutil.copy(os.path.join(src, filename),
                            os.path.join(dst, filename))
            else:
                raise
