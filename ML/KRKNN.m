//
//  KRKNN.m
//  KRKNN
//
//  Created by Kalvar Lin on 2015/8/30.
//  Copyright (c) 2015年 Kalvar Lin. All rights reserved.
//

#import "KRKNN.h"
#import "KRKNN+Definitions.h"

@interface KRKNN ()

@end

@implementation KRKNN (fixCalculations)

// Calculated by Cosine Similarity method, 歸屬度概念是越大越近
-(float)_distanceCosineWithClassifiedFeatures:(NSArray *)_classifiedFeatures patternFeatures:(NSArray *)_patternFeatures
{
    float _sumA  = 0.0f;
    float _sumB  = 0.0f;
    float _sumAB = 0.0f;
    int _index   = 0;
    for( NSNumber *_featureValue in _classifiedFeatures )
    {
        NSNumber *_patternValue = [_patternFeatures objectAtIndex:_index];
        float _aValue  = [_featureValue floatValue];
        float _bValue  = [_patternValue floatValue];
        _sumA         += ( _aValue * _aValue );
        _sumB         += ( _bValue * _bValue );
        _sumAB        += ( _aValue * _bValue );
        ++_index;
    }
    float _ab = _sumA * _sumB;
    return ( _ab > 0.0f ) ? ( _sumAB / sqrtf( _ab ) ) : 0.0f;
}

// Euclidean distance which multi-dimensional formula, 距離概念是越小越近
-(float)_distanceEuclideanWithClassifiedFeatures:(NSArray *)_classifiedFeatures patternFeatures:(NSArray *)_patternFeatures
{
    NSInteger _index = 0;
    float _sum       = 0.0f;
    for( NSNumber *_x in _classifiedFeatures )
    {
        _sum        += powf([_x floatValue] - [[_patternFeatures objectAtIndex:_index] floatValue], 2);
        ++_index;
    }
    return (_index > 0) ? sqrtf(_sum) : _sum;
}

// 距離概念是越小越近, 歸屬度概念是越大越近 (也能用 1.0f 取其差值，即能越小越近)
-(float)_distanceWithClassifiedFeatures:(NSArray *)_classifiedFeatures patternFeatures:(NSArray *)_patternFeatures
{
    float _distance = 0.0f;
    switch (self.kernel)
    {
        case KRKNNKernelByCosineSimilarity:
            _distance = 1.0f - [self _distanceCosineWithClassifiedFeatures:_classifiedFeatures patternFeatures:_patternFeatures];
            break;
        case KRKNNKernelByEuclidean:
            _distance = [self _distanceEuclideanWithClassifiedFeatures:_classifiedFeatures patternFeatures:_patternFeatures];
            break;
        default:
            break;
    }
    //NSLog(@"_distance : %f", _distance);
    return _distance;
}

-(void)_addClassifiedSetsAtFeatures:(NSArray *)_features group:(NSString *)_group identifier:(NSString *)_identifier
{
    [self.trainingSets setObject:[_features copy] forKey:_identifier];
    [self.trainingGroups setObject:[_group copy] forKey:_identifier];
    [self.allData setObject:@[_group, _identifier, _features] forKey:_identifier];
}

@end

@implementation KRKNN

+(instancetype)sharedInstance
{
    static dispatch_once_t pred;
    static KRKNN *_object = nil;
    dispatch_once(&pred, ^
    {
        _object = [[KRKNN alloc] init];
    });
    return _object;
}

-(instancetype)init
{
    self = [super init];
    if( self )
    {
        _trainingSets   = [NSMutableDictionary new];
        _trainingGroups = [NSMutableDictionary new];
        _allData        = [NSMutableDictionary new];
        _kernel         = KRKNNKernelByCosineSimilarity;
    }
    return self;
}

#pragma --mark Creates Training Sets
// Adding the training sets be the basic patterns to do classfication.
// @param _group means the training-sets own to which group.
-(void)addFeatures:(NSArray *)_features group:(NSString *)_group identifier:(NSString *)_identifier
{
    if( nil == _features || [_features count] < 1 )
    {
        return;
    }
    
    if( nil == _group || [_group length] < 1 )
    {
        _group = unknown;
    }
    
    if( nil == _identifier || [_identifier length] < 1 )
    {
        _identifier = [NSString stringWithFormat:@"%@%li", unknown, [_trainingSets count]];
    }
    
    // Uses key/value to make the basic training-sets, one key for one point on the hyperplane of classification.
    [self _addClassifiedSetsAtFeatures:_features
                                 group:_group
                            identifier:_identifier];
}

-(void)classifyFeatures:(NSArray *)_features identifier:(NSString *)_identifier kNeighbor:(NSInteger)_kNeighbor completion:(KRKNNCompletion)_doCompletion
{
    if( nil == _features || [_features count] < 1 )
    {
        return;
    }
    
    if( nil == _trainingSets || [_trainingSets count] < 1 )
    {
        return;
    }
    
    if( nil == _identifier || [_identifier length] < 1 )
    {
        _identifier = [NSString stringWithFormat:@"%@%li", unknown, [_trainingSets count] + 1];
    }
    
    NSMutableArray *_sorts = [NSMutableArray new];
    NSString *_idKey       = identifierKey;
    NSString *_distanceKey = distanceKey;
    // Catchs every feature
    for( NSString *_classifiedId in _trainingSets )
    {
        float _distance = [self _distanceWithClassifiedFeatures:[_trainingSets objectForKey:_classifiedId] patternFeatures:_features];
        [_sorts addObject:@{_idKey : _classifiedId, _distanceKey : [NSNumber numberWithFloat:_distance]}];
    }
    
    NSSortDescriptor *_sortDescriptor    = [NSSortDescriptor sortDescriptorWithKey:_distanceKey ascending:YES];
    NSArray *_sortedGroups               = [_sorts sortedArrayUsingDescriptors:[NSArray arrayWithObject:_sortDescriptor]];
    
    // Catchs K neighbors
    NSMutableDictionary *_countingNears  = [NSMutableDictionary new];
    NSMutableDictionary *_sumDistances   = [NSMutableDictionary new];
    NSInteger _maxCounting               = 0;
    NSString *_ownGroup                  = @"";
    BOOL _success                        = NO;
    NSInteger _k                         = 0;
    for( NSDictionary *_neighbors in _sortedGroups )
    {
        // Catchs classification group by pattern id
        NSString *_classifiedGroup = [_trainingGroups objectForKey:[_neighbors objectForKey:_idKey]];
        // Counting how many group types nearby the pattern
        NSNumber *_times           = [_countingNears objectForKey:_classifiedGroup];
        NSInteger _counting        = 1;
        if( nil != _times )
        {
            _counting += [_times integerValue];
        }
        
        if( _counting > _maxCounting )
        {
            _maxCounting = _counting;
            _ownGroup    = _classifiedGroup;
        }
        [_countingNears setObject:[NSNumber numberWithInteger:_counting] forKey:_classifiedGroup];
        
        // Sum the distance of neighbor of pattern, 計算同群的鄰居們總距離 (用於輔助判斷在群組鄰居數相同時，最後該被分到哪一群裡)
        NSNumber *_groupDistance    = [_sumDistances objectForKey:_classifiedGroup];
        NSNumber *_neighborDistance = [_neighbors objectForKey:_distanceKey];
        float _distance             = 0.0f;
        if( nil != _groupDistance )
        {
            _distance = [_groupDistance floatValue];
        }
        _distance += [_neighborDistance floatValue];
        [_sumDistances setObject:[NSNumber numberWithFloat:_distance] forKey:_classifiedGroup];
        
        ++_k;
        if( _k >= _kNeighbor )
        {
            /*
             * @ Notes
             *   - 判斷所屬群組的最大鄰居數是否大於最小所需可判斷被正確分群的鄰居數，例如，有 3 個群，那 k 數量最保險會需要從 4 個鄰居裡去統計離哪群最近，
             *     如果 < 4，有可能會發生每群都是相等的最近鄰居數量 (例如 1 群 1 個)，只要 k 個鄰居數量是能被群組數量 % 整除的，就有可能會有這問題。
             */
            // 檢查統計好的分群，是否有跟其它分群的鄰居數量相等的情況
            float _ownDistance = [[_sumDistances objectForKey:_ownGroup] floatValue];
            NSString *_myGroup = [_ownGroup copy];
            for( NSString *_groupName in _countingNears )
            {
                if( [_groupName isEqualToString:_myGroup] )
                {
                    continue;
                }
                // 如果其它群有跟分到的群相同的最近鄰居數，則需比較每群距離來決定該把 Pattern 分到哪群 (挑最小距離總和)
                if( [[_countingNears objectForKey:_groupName] floatValue] == _maxCounting )
                {
                    float _otherDistance = [[_sumDistances objectForKey:_groupName] floatValue];
                    if( _otherDistance < _ownDistance )
                    {
                        _ownDistance = _otherDistance;
                        _ownGroup    = _groupName;
                    }
                }
            }
            break;
        }
    }
    
    if( _maxCounting > 0 )
    {
        _success = YES;
        [self _addClassifiedSetsAtFeatures:_features
                                     group:_ownGroup
                                identifier:_identifier];
    }
    
    if( _doCompletion )
    {
        _doCompletion(_success, _ownGroup, _maxCounting, _allData);
    }
    
}

@end