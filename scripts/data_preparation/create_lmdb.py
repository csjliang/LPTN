from codes.utils import scandir
from codes.utils.lmdb_util import make_lmdb_from_imgs


def create_lmdb():

    folder_path = 'datasets/FiveK/FiveK_480p/train/A'
    lmdb_path = 'datasets/FiveK/FiveK_train_source.lmdb'
    img_path_list, keys = prepare_keys(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = 'datasets/FiveK/FiveK_480p/train/B'
    lmdb_path = 'datasets/FiveK/FiveK_train_target.lmdb'
    img_path_list, keys = prepare_keys(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = 'datasets/FiveK/FiveK_480p/test/A'
    lmdb_path = 'datasets/FiveK/FiveK_test_source.lmdb'
    img_path_list, keys = prepare_keys(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

    folder_path = 'datasets/FiveK/FiveK_480p/test/B'
    lmdb_path = 'datasets/FiveK/FiveK_test_target.lmdb'
    img_path_list, keys = prepare_keys(folder_path)
    make_lmdb_from_imgs(folder_path, lmdb_path, img_path_list, keys)

def prepare_keys(folder_path):

    print('Reading image path list ...')
    img_path_list = sorted(
        list(scandir(folder_path, suffix='jpg', recursive=False)))
    keys = [img_path.split('.jpg')[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys


if __name__ == '__main__':

    create_lmdb()

