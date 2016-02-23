//
//  KRKNN.m
//  KRKNN
//
//  Created by Kalvar Lin on 2015/8/30.
//  Copyright (c) 2015年 Kalvar Lin. All rights reserved.
//

#import "KRKNN.h"

@interface KRKNN ()

@property (nonatomic, strong) KRKNNPatternMutableDictionary *trainingSets;

@end

#pragma mark - KRKNN Category fixCalculations

@implementation KRKNN (fixCalculations)

// Calculated by Cosine Similarity method, 歸屬度概念是越大越近
-(float)_cosineWithClassifiedFeatures:(NumberArray *)_classifiedFeatures patternFeatures:(NumberArray *)_patternFeatures
{
    float _sumA        = 0.0f;
    float _sumB        = 0.0f;
    float _sumAB       = 0.0f;
    for( int _index = 0; _index < _classifiedFeatures.count; _index++ )
    {
        float _aValue  = _classifiedFeatures[_index].floatValue;
        float _bValue  = _patternFeatures[_index].floatValue;
        _sumA         += powf( _aValue, 2 );
        _sumB         += powf( _bValue, 2 );
        _sumAB        += ( _aValue * _bValue );
    }
    
    // 由於 _sumA 與 _sumB 皆為平方值疊加, 因此 _ab 必 >= 0
    float _ab = _sumA * _sumB;
    return _ab ? ( _sumAB / sqrtf( _ab ) ) : 0.0f;
}

// Euclidean distance which multi-dimensional formula, 距離概念是越小越近
-(float)_euclideanWithClassifiedFeatures:(NumberArray *)_classifiedFeatures patternFeatures:(NumberArray *)_patternFeatures
{
    float _sum       = 0.0f;
    
    // 由於 _sum 加的皆是平方值, 因此 _sum 必 >= 0
    for( int _index = 0; _index < _classifiedFeatures.count; _index++ )
    {
        _sum        += powf( _classifiedFeatures[_index].floatValue - _patternFeatures[_index].floatValue, 2 );
    }
    return sqrtf( _sum );
}

-(double)_rbf:(NumberArray *)_x1 x2:(NumberArray *)_x2 sigma:(float)_sigma
{
    double _sum      = 0.0f;
    for( int _index = 0; _index < _x1.count; _index++ )
    {
        // Formula : s = s + ( v1[i] - v2[i] )^2
        double _v    = _x1[_index].doubleValue - _x2[_index].doubleValue;
        _sum        += pow( _v, 2 );
    }
    
    // Formula : exp^( -s / ( 2.0f * sigma * sigma ) )
    return pow( M_E, ( -_sum ) / ( 2.0f * _sigma * _sigma ) );
}

// 距離概念是越小越近, 歸屬度概念是越大越近 (也能用 1.0f 取其差值，即能越小越近)
-(float)_distanceWithClassifiedFeatures:(NumberArray *)_classifiedFeatures patternFeatures:(NumberArray *)_patternFeatures
{
    float _distance = 0.0f;
    switch ( self.kernel )
    {
        case KRKNNKernelCosineSimilarity:
            _distance = 1.0f - [self _cosineWithClassifiedFeatures:_classifiedFeatures patternFeatures:_patternFeatures];
            break;
        case KRKNNKernelEuclidean:
            _distance = [self _euclideanWithClassifiedFeatures:_classifiedFeatures patternFeatures:_patternFeatures];
            break;
        case KRKNNKernelRBF:
            _distance = [self _rbf:_classifiedFeatures x2:_patternFeatures sigma:2.0f];
            break;
        default:
            break;
    }
    //NSLog(@"_distance : %f", _distance);
    return _distance;
}

-(void)_addClassifiedSetsAtFeatures:(NumberArray *)_features groupName:(NSString *)_group identifier:(NSString *)_identifier
{
    [self.trainingSets setObject:[[KRKNNPattern alloc] initWithFeatures:_features groupName:_group identifier:_identifier]
                          forKey:_identifier];
}

@end

#pragma mark - Class KRKNN

@implementation KRKNN

#pragma mark - Class Method

+(instancetype)sharedInstance
{
    static dispatch_once_t pred;
    static KRKNN *_object = nil;
    dispatch_once(&pred, ^{
        _object = [KRKNN new];
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
        _trainingSets = [NSMutableDictionary dictionary];
        _kernel       = KRKNNKernelCosineSimilarity;
    }
    return self;
}

#pragma mark * Creates Training Sets

// Adding the training sets be the basic patterns to do classfication.
// @param groupName means the training-sets own to which group.
-(void)addFeatures:(NumberArray *)_features groupName:(NSString *)_groupName identifier:(NSString *)_identifier
{
    if( nil == _features || _features.count < 1 )
    {
        return;
    }
    
    if( nil == _groupName || _groupName.length < 1 )
    {
        _groupName = kUnknownStatus;
    }
    
    if( nil == _identifier || _identifier.length < 1 )
    {
        _identifier = [NSString stringWithFormat:@"%@%td", kUnknownStatus, _trainingSets.count];
    }
    
    // Uses key/value to make the basic training-sets, one key for one point on the hyperplane of classification.
    [self _addClassifiedSetsAtFeatures:_features
                             groupName:_groupName
                            identifier:_identifier];
}

-(void)classifyFeatures:(NumberArray *)_features identifier:(NSString *)_identifier kNeighbor:(NSInteger)_kNeighbor completion:(KRKNNCompletion)_doCompletion
{
    if( nil == _features || _features.count < 1 )
    {
        return;
    }
    
    if( nil == _trainingSets || _trainingSets.count < 1 )
    {
        return;
    }
    
    if( nil == _identifier || _identifier.length < 1 )
    {
        _identifier = [NSString stringWithFormat:@"%@%td", kUnknownStatus, _trainingSets.count + 1];
    }
    
    NSMutableArray *_sorts = [NSMutableArray array];
    NSString *_idKey       = kIdentifierKey;
    NSString *_distanceKey = kDistanceKey;
    
    // Catchs every feature
    for( NSString *_classifiedId in _trainingSets )
    {
        KRKNNPattern *_pattern = _trainingSets[_classifiedId];
        float _distance = [self _distanceWithClassifiedFeatures:_pattern.features patternFeatures:_features];
        [_sorts addObject:@{_idKey : _classifiedId, _distanceKey : [NSNumber numberWithFloat:_distance]}];
    }
    
    NSSortDescriptor *_sortDescriptor    = [NSSortDescriptor sortDescriptorWithKey:_distanceKey ascending:YES];
    NSArray *_sortedGroups               = [_sorts sortedArrayUsingDescriptors:@[_sortDescriptor]];
    
    // Catchs K neighbors
    NSNumberMutableDictionary *_countingNears  = [NSMutableDictionary dictionary];
    NSNumberMutableDictionary *_sumDistances   = [NSMutableDictionary dictionary];
    NSInteger _maxCounting                     = 0;
    NSString *_ownGroup                        = @"";
    NSInteger _k                               = 0;
    for( NSDictionary *_neighbors in _sortedGroups )
    {
        // Catchs classification group by pattern id, 取出該鄰居是屬於哪一群
        NSString *_classifiedGroup = _trainingSets[_neighbors[_idKey]].groupName;
        
        // Counting how many group types nearby the pattern
        NSNumber *_times           = _countingNears[_classifiedGroup];
        NSInteger _counting        = 1;
        if( nil != _times )
        {
            _counting += _times.integerValue;
        }
        
        if( _counting > _maxCounting )
        {
            _maxCounting = _counting;
            _ownGroup    = _classifiedGroup;
        }
        _countingNears[_classifiedGroup] = @(_counting);
        
        // Sum the distance of neighbor of pattern, 計算同群的鄰居們總距離 (用於輔助判斷在群組鄰居數相同時，最後該被分到哪一群裡)
        NSNumber *_groupDistance    = _sumDistances[_classifiedGroup];
        NSNumber *_neighborDistance = _neighbors[_distanceKey];
        float _distance             = 0.0f;
        if( nil != _groupDistance )
        {
            _distance = _groupDistance.floatValue;
        }
        _distance += _neighborDistance.floatValue;
        _sumDistances[_classifiedGroup] = @(_distance);
        
        ++_k;
        if( _k >= _kNeighbor )
        {
            /*
             * @ Notes
             *   - 判斷所屬群組的最大鄰居數是否大於最小所需可判斷被正確分群的鄰居數，例如，有 3 個群，那 k 數量最保險會需要從 4 個鄰居裡去統計離哪群最近，
             *     如果 < 4，有可能會發生每群都是相等的最近鄰居數量 (例如 1 群 1 個)，只要 k 個鄰居數量是能被群組數量 % 整除的，就有可能會有這問題。
             */
            // 檢查統計好的分群，是否有跟其它分群的鄰居數量相等的情況
            float _ownDistance = _sumDistances[_ownGroup].floatValue;
            NSString *_myGroup = [_ownGroup copy];
            for( NSString *_groupName in _countingNears )
            {
                if( [_groupName isEqualToString:_myGroup] )
                {
                    continue;
                }
                
                // 如果其它群有跟分到的群相同的最近鄰居數，則需比較每群距離來決定該把 Pattern 分到哪群 (挑最小距離總和)
                if( _countingNears[_groupName].floatValue == _maxCounting )
                {
                    float _otherDistance = _sumDistances[_groupName].floatValue;
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
    
    BOOL _success = ( _maxCounting > 0 );
    if( _success )
    {
        [self _addClassifiedSetsAtFeatures:_features
                                 groupName:_ownGroup
                                identifier:_identifier];
    }
    
    if( _doCompletion )
    {
        _doCompletion(_success, _ownGroup, _maxCounting, [_trainingSets allValues]);
    }
    
}

-(void)classifyFeatures:(NumberArray *)_features identifier:(NSString *)_identifier completion:(KRKNNCompletion)_doCompletion
{
    [self classifyFeatures:_features identifier:_identifier kNeighbor:[self chooseK] completion:_doCompletion];
}

// Use sqrt(N) to quickly and simply choose the K number.
-(NSInteger)chooseK
{
    return (NSInteger)ceilf( sqrtf( [_trainingSets count] ) );
}

@end