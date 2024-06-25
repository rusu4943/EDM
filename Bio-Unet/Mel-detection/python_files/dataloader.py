import os
import h5py
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

def get_transform():
    img_size = ( 512, 512 )
    transform = transforms.Compose( [
        transforms.Resize( img_size ),
        # transforms.CenterCrop( (224, 224) ), 
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225)) ] )
    return transform

class SkinDataset( torch.utils.data.Dataset ):

    def __init__( self, h5_file_path, transform ):
        self.h5_file_path = h5_file_path
        self.h5_file = None
        self.transform = transform
        self.img_id_to_h5idx = self.build_img_id_to_h5idx()
        self.num_imgs = self.get_num_imgs() 

    def get_num_imgs( self ):
        with h5py.File( self.h5_file_path, 'r' ) as f:
            return len( f['image_ids'] )

    def build_img_id_to_h5idx( self ):
        with h5py.File( self.h5_file_path, 'r' ) as f:
            img_ids = f['image_ids']
            img_id_to_h5idx = \
                    { img_id : idx for idx, img_id in enumerate( img_ids ) }
            return img_id_to_h5idx

    def __len__( self ):
        return self.get_num_imgs()

    def __getitem__( self, idx ):
        # import pdb; pdb.set_trace()
        if not self.h5_file:
            self.h5_file = h5py.File( self.h5_file_path, 'r' )
        img_id = self.h5_file['image_ids'][idx]
        img = self.h5_file['images'][idx]
        masks = self.h5_file['masks'][idx]
        labels = self.h5_file['labels'][idx].astype(np.float)
        img = img.transpose([1, 2, 0])
        img = Image.fromarray( np.uint8(img) )
        if self.transform:
            img = self.transform( img )

        return img_id, img, labels[5], masks

# only for testing
class Args( object ):

    def __init__( self, input_dir, batch_size,
            num_workers ):
        self.input_dir = input_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

def get_dataloader( args ):
    dataloader = {}
    splits = [ 'train', 'val' ]
    for split in splits:
        h5_file_path = os.path.join( args.input_dir1, f'{split}.h5' )
        transform = get_transform()
        dataset = SkinDataset( h5_file_path, transform )
        loader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=args.batch_size,
                shuffle=True if split == 'train' else False,
                num_workers=args.num_workers, )
        dataloader[ split ] = loader
    return dataloader

def _test():
    input_dir = './data/'
    batch_size = 8
    num_workers = 0
    args = Args( input_dir, batch_size, num_workers )
    import pdb; pdb.set_trace()
    dataloader = get_dataloader( args )[ 'val' ]
    for step, ( img_id, img, labels, masks ) in enumerate( dataloader ):
        assert img_id.shape == ( batch_size, )
        assert img.shape == ( batch_size, 3, 512, 512 )
        assert labels.shape == ( batch_size, 5 )
        assert masks.shape == ( batch_size, 5, 512, 512 )
        if step >= 5: break
    print( 'Test Passed!' )

if __name__ == '__main__':
    _test()
