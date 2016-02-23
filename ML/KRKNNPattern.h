//
//  KRKNNPattern.h
//  KRKNN
//
//  Created by Kalvar Lin on 2016/2/22.
//  Copyright © 2016年 Kalvar Lin. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "KRKNN+Definitions.h"

@interface KRKNNPattern : NSObject

@property (nonatomic, strong, readonly) NumberMutableArray *features;
@property (nonatomic, strong, readonly) NSString *groupName;
@property (nonatomic, strong, readonly) NSString *identifier;

+(instancetype)sharedPattern;
-(instancetype)init;
-(instancetype)initWithFeatures:(NumberArray *)_kFeatures groupName:(NSString *)_kGroupName identifier:(NSString *)_kIdentifier;

-(void)addFeatures:(NumberArray *)_kFeatures groupName:(NSString *)_kGroupName identifier:(NSString *)_kIdentifier;

@end
