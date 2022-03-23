from data.DIT import dump_data as dump_data_image
from data.DIT import collect_data as collect_data_image
from data.DIT import load_data as load_data_image

if __name__ == '__main__':
    dump_data_image('data/superblue7_processed', 'data/superblue7', 800, 8, '100000', force_save=True)
    dump_data_image('data/superblue9_processed', 'data/superblue9', 800, 8, '100000', force_save=True)
    dump_data_image('data/superblue14_processed', 'data/superblue14', 800, 8, '100000', force_save=True)
    dump_data_image('data/superblue16_processed', 'data/superblue16', 700, 8, '100000', force_save=True)
    dump_data_image('data/superblue19_processed', 'data/superblue19', 900, 8, '100000', force_save=True)
    collect_data_image([
        'superblue1_processed',
        'superblue2_processed',
        'superblue3_processed',
        'superblue5_processed',
        'superblue6_processed',
        'superblue7_processed',
        'superblue9_processed',
        'superblue11_processed',
        'superblue14_processed',
    ], 'data/train_images', clear_files=True)
    collect_data_image([
        # 'superblue16_processed',
        'superblue19_processed',
    ], 'data/test_images', clear_files=False)
    _, train_images = load_data_image('data/train_images')
    _, test_images = load_data_image('data/test_images')
    print(len(train_images))
    print(len(test_images))
