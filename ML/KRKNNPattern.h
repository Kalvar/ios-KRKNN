//
//  KRKNNPattern.h
//  KRKNN
//
//  Created by Kalvar Lin on 2016/2/22.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import <Foundation/Foundation.h>

@interface KRKNNPattern : NSObject

@property (nonatomic, strong) NSMutableArray *features;
@property (nonatomic, strong) NSString *groupName;
@property (nonatomic, strong) NSString *identifier;

+(instancetype)sharedPattern;
-(instancetype)init;
-(instancetype)initWithFeatures:(NSArray *)_kFeatures groupName:(NSString *)_kGroupName identifier:(NSString *)_kIdentifier;

-(void)addFeatures:(NSArray *)_kFeatures groupName:(NSString *)_kGroupName identifier:(NSString *)_kIdentifier;

@end
