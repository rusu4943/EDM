import torch

import sys
sys.path.append('../../')
import python_files.models as models
#import models

def get_model( args ):
    backbone = models.get_backbone( args )
    model = models.BirdClassifier(
            backbone=backbone,
            num_classes=args.num_classes,
            cls_type=args.cls_type,
            hidden_dim=args.hidden_dim )
    return model

def get_optimizer( args, model ):
    optimizer = torch.optim.Adam(
            params=filter( lambda p: p.requires_grad, model.parameters() ),
            lr=args.learn_rate,
            weight_decay=args.weight_decay )
    return optimizer

