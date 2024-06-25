import torch
import torch.nn as nn
import random
from torchvision import models

class Backbone( nn.Module ):

    def __init__( self, mode='fine-tune' ):
        super( Backbone, self ).__init__()
        assert mode in [ 'fine-tune', 'feature-extract' ]
        self.mode = mode

    def forward( self, x ):
        raise NotImplementedError()
    

class Resnet50( Backbone ):

    def __init__( self, mode='fine-tune' ):
        super( Resnet50, self ).__init__( mode )
        resnet = models.resnet50( pretrained=True )
        self.out_ch = resnet.fc.in_features
        # remove fc layer from resnet
        resnet = list(resnet.children())[:-1]
        self.model = nn.Sequential( *resnet )
        self.fine_tune( self.mode == 'fine-tune' )

    def forward( self, x ):
        out = self.model( x )
        return out.squeeze()
    
    def fine_tune( self, tune=True ):
        '''
        Enable fine-tuning weights if tune=True
        '''
        for param in self.model.parameters():
            param.requires_grad = False
        
        # enable weight updates for layers 2 to 4
        for child in list( self.model.children() )[ -4: ]:
            for param in child.parameters():
                param.requires_grad = tune

class Vgg19( Backbone ):
    
    def __init__( self, mode='fine-tune' ):
        super( Vgg19, self ).__init__( mode )
        vgg = models.vgg19( pretrained=True )
        self.out_ch = vgg.classifier[-1].in_features
        # remove last fc layer
        vgg.classifier = nn.Sequential(
            *list(vgg.classifier.children())[:-1] )
        self.model = vgg
        self.fine_tune( self.mode == 'fine-tune' )
    
    def forward( self, x ):
        out = self.model( x )
        return out.squeeze()

    def fine_tune( self, tune=True ):
        # import pdb; pdb.set_trace()
        for param in self.model.parameters():
            param.requires_grad = False

        # enable weight updates
        for child in self.model.classifier.children():
            for param in child.parameters():
                param.requires_grad = tune

        for i, child in enumerate( self.model.features.children() ):
            if i < 36: continue
            for param in child.parameters():
                param.requires_grad = tune

class InceptionV3( Backbone ):

    def __init__( self, mode='fine-tune' ):
        super( InceptionV3, self ).__init__()
        inv3 = models.inception_v3( pretrained=True )
        self.out_ch = inv3.fc.in_features
        self.aux_out_ch = inv3.AuxLogits.fc.in_features
        inv3.AuxLogits.fc = nn.Identity()
        inv3.fc = nn.Identity()
        self.model = inv3
        self.fine_tune( self.mode == 'fine-tune' )

    def forward( self, x ):
        out = self.model( x )[ 0 ]
        return out

    def fine_tune( self, tune=True ):
        for param in self.model.parameters():
            param.requires_grad = False

        # enable weight updates
        for param in self.model.Mixed_7c.parameters():
            param.requires_grad = tune

def get_backbone( args ):
    assert args.backbone in [ 'resnet50', 'vgg19', 'inception_v3' ]
    backbone = None
    if args.backbone == 'resnet50':
        backbone = Resnet50( args.tune_mode )
    elif args.backbone == 'vgg19':
        backbone = Vgg19( args.tune_mode )
    elif args.backbone == 'inception_v3':
        backbone = InceptionV3( args.tune_mode )
    else:
        raise Exception( 'unrecognized backbone' )
    return backbone


def get_cls_arch( cls_type, in_dim, out_dim, hidden_dim ):
    assert cls_type in [ 'single', 'double', 'double-bn',
            'double-dropout' ]
    cls = None
    if cls_type == 'single':
        cls = ClsSingleLayer( in_dim, out_dim )
    elif cls_type == 'double':
        cls = ClsDoubleLayer( in_dim, out_dim, hidden_dim )
    elif cls_type == 'double-bn':
        cls = ClsDoubleLayerBn( in_dim, out_dim, hidden_dim )
    elif cls_type == 'double-dropout':
        cls = ClsDoubleLayerDropout( in_dim, out_dim, hidden_dim )
    else:
        raise Exception( 'unrecognized architecture type: ' + cls_type )
    return cls

class ClsSingleLayer( nn.Module ):

    def __init__( self, in_dim, out_dim ):
        super( ClsSingleLayer, self ).__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward( self, x ):
        out = self.fc( x )
        return out

class ClsDoubleLayer( nn.Module ):

    def __init__( self, in_dim, out_dim, hidden_dim=512 ):
        super( ClsDoubleLayer, self ).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()

    def forward( self, x ):
        x = self.relu( self.fc1( x ) )
        out = self.fc2( x )
        return out

class ClsDoubleLayerBn( nn.Module ):

    def __init__( self, in_dim, out_dim, hidden_dim=512 ):
        super( ClsDoubleLayerBn, self ).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward( self, x ):
        x = self.relu( self.bn( self.fc1( x ) ) )
        out = self.fc2( x )
        return out

class ClsDoubleLayerDropout( nn.Module ):

    def __init__( self, in_dim, out_dim, hidden_dim=512 ):
        super( ClsDoubleLayerDropout, self ).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.5)

    def forward( self, x ):
        x = self.relu( self.bn( self.fc1( x ) ) )
        x = self.dropout( x )
        out = self.fc2( x )
        return out

class BirdClassifier( nn.Module ):

    def __init__( self, backbone, num_classes, cls_type='single',
            hidden_dim=512 ):
        super( BirdClassifier, self ).__init__()
        assert isinstance( backbone, Backbone )
        self.num_classes = num_classes
        self.backbone = backbone
        self.cls = get_cls_arch( cls_type, backbone.out_ch, num_classes,
                hidden_dim )

    def forward( self, x ):
        '''
        forward method
        x: input img tensor
        '''
        # import pdb; pdb.set_trace()
        x = self.backbone( x )
        out = self.cls( x )
        # hack to make things work for batch_size 1
        out = out.unsqueeze(dim=0) if out.dim() == 1 else out
        return out

class Args( object ):
    def __init__( self, mode, backbone ):
        self.tune_mode = mode
        self.backbone = backbone

def _test_model( backbone, cls_type, hidden_dim ):
    # import pdb; pdb.set_trace()
    num_classes = 28
    mode = 'fine-tune'
    img_size = 299 if backbone == 'inception_v3' else 224
    batch_size = 2
    loss_fn = nn.CrossEntropyLoss()
    args = Args( mode, backbone )
    backbone = get_backbone( args )
    bc = BirdClassifier( backbone, num_classes, cls_type, hidden_dim )
    img = torch.randn( batch_size, 3, img_size, img_size )
    labels = torch.randint( num_classes, (batch_size,) )
    pred = bc( img )
    assert pred.shape == ( batch_size, num_classes )
    loss = loss_fn( pred, labels )

def _test():
    backbones = [ 'resnet50', 'vgg19', 'inception_v3' ]
    cls_types = [ 'single', 'double', 'double-bn', 'double-dropout' ]
    # cls_types = ['double-dropout']
    hidden_dims = [64, 128, 256, 512]
    for backbone in backbones:
        for cls_type in cls_types:
            hidden_dim = random.choice( hidden_dims )
            _test_model( backbone=backbone, cls_type=cls_type,
                    hidden_dim=hidden_dim )
    print( 'Test Passed!' )

if __name__ == '__main__':
    _test()
