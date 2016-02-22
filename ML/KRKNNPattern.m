//
//  KRKNNPattern.m
//  KRKNN
//
//  Created by Kalvar Lin on 2016/2/22.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRKNNPattern.h"

@implementation KRKNNPattern

+(instancetype)sharedPattern
{
    static dispatch_once_t pred;
    static KRKNNPattern *_object = nil;
    dispatch_once(&pred, ^{
        _object = [[KRKNNPattern alloc] init];
    });
    return _object;
}

-(instancetype)init
{
    self = [super init];
    if( self )
    {
        _features   = [NSMutableArray new];
        _groupName  = @"";
        _identifier = @"";
    }
    return self;
}

-(void)addFeatures:(NSArray *)_kFeatures groupName:(NSString *)_kGroupName identifier:(NSString *)_kIdentifier
{
    [_features addObjectsFromArray:[_kFeatures copy]];
    _groupName  = _kGroupName;
    _identifier = _kIdentifier;
}

@end
