//
//  KRKNN.h
//  KRKNN
//
//  Created by Kalvar Lin on 2015/8/30.
//  Copyright (c) 2015年 Kalvar Lin. All rights reserved.
//

#import <Foundation/Foundation.h>
#import "KRKNN+Definitions.h"
#import "KRKNNPattern.h"

typedef enum KRKNNKernels
{
    KRKNNKernelCosineSimilarity = 0,
    KRKNNKernelEuclidean,
    KRKNNKernelRBF
}KRKNNKernels;

typedef void(^KRKNNCompletion)(BOOL success, NSString *ownGroup, NSInteger neighborCount, NSArray *allPatterns);

@interface KRKNN : NSObject

@property (nonatomic, assign) KRKNNKernels kernel;

+(instancetype)sharedInstance;
-(instancetype)init;

-(void)addFeatures:(NumberArray *)_features groupName:(NSString *)_group identifier:(NSString *)_identifier;
-(void)classifyFeatures:(NumberArray *)_features identifier:(NSString *)_identifier kNeighbor:(NSInteger)_kNeighbor completion:(KRKNNCompletion)_doCompletion;
-(void)classifyFeatures:(NumberArray *)_features identifier:(NSString *)_identifier completion:(KRKNNCompletion)_doCompletion;

-(NSInteger)chooseK;


@end