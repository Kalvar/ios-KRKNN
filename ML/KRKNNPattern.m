//
//  KRKNNPattern.m
//  KRKNN
//
//  Created by Kalvar Lin on 2016/2/22.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import "KRKNNPattern.h"

#pragma mark - Class KRKNNPattern

@implementation KRKNNPattern

#pragma mark - Class Method

+(instancetype)sharedPattern
{
    static dispatch_once_t pred;
    static KRKNNPattern *_object = nil;
    dispatch_once(&pred, ^{
        _object = [KRKNNPattern new];
    });
    return _object;
}

#pragma mark - Instance Method

#pragma mark * Init

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

-(instancetype)initWithFeatures:(NumberArray *)_kFeatures groupName:(NSString *)_kGroupName identifier:(NSString *)_kIdentifier
{
    self = [self init];
    if( self )
    {
        [self addFeatures:_kFeatures groupName:_kGroupName identifier:_kIdentifier];
    }
    return self;
}

#pragma mark * Modify Variables

-(void)addFeatures:(NumberArray *)_kFeatures groupName:(NSString *)_kGroupName identifier:(NSString *)_kIdentifier
{
    [_features addObjectsFromArray:[_kFeatures copy]];
    _groupName  = _kGroupName;
    _identifier = _kIdentifier;
}

@end
