//
//  KRKNN.h
//  KRKNN
//
//  Created by Kalvar Lin on 2015/8/30.
//  Copyright (c) 2015年 Kalvar Lin. All rights reserved.
//

#import <Foundation/Foundation.h>

typedef enum KRKNNKernels
{
    KRKNNKernelByCosineSimilarity = 0,
    KRKNNKernelByEuclidean        = 1
}KRKNNKernels;

typedef void(^KRKNNCompletion)(BOOL success, NSString *ownGroup, NSInteger neighborCount, NSDictionary *allData);

@interface KRKNN : NSObject

@property (nonatomic, strong) NSMutableDictionary *allData;
@property (nonatomic, strong) NSMutableDictionary *trainingSets;
@property (nonatomic, strong) NSMutableDictionary *trainingGroups;
@property (nonatomic, assign) KRKNNKernels kernel;
@property (nonatomic, copy) KRKNNCompletion completion;

+(instancetype)sharedInstance;
-(instancetype)init;

-(void)addFeatures:(NSArray *)_features group:(NSString *)_group identifier:(NSString *)_identifier;
-(void)classifyFeatures:(NSArray *)_features identifier:(NSString *)_identifier kNeighbor:(NSInteger)_kNeighbor completion:(KRKNNCompletion)_doCompletion;


@end