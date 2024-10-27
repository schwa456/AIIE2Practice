import numpy as np
from sklearn.metrics import mutual_info_score

def mRMR(X, y, num_features):
    """
    mRMR 알고리즘을 사용하여 특징을 선택하는 함수

    Parameters:
    - X: 2차원 numpy 배열 (샘플 수 x 특징 수) - 입력 특징 행렬
    - y: 1차원 numpy 배열 또는 리스트 - 대상 변수 (레이블)
    - num_features: 선택할 특징의 수

    Returns:
    - selected_features: 선택된 특징의 인덱스 리스트
    """

    # 변수 초기화
    n_features = X.shape[1]
    selected_features = []
    remaining_features = list(range(n_features))

    # 상호정보량 계산
    relevance = np.array([mutual_info_score(X[:, i], y) for i in range(n_features)])

    # 첫번째 feature 선택(최대 관련성 기준)
    first_feature = np.argmax(relevance)
    selected_features.append(first_feature)
    remaining_features.remove(first_feature)

    # 남은 특징에서 반복적으로 특징 선택
    for _ in range(num_features - 1):
        max_score = -np.inf
        best_feature = -1

        # 각 남은 특징에 대해 mRMR 계산
        for feature in remaining_features:
            redundancy = np.mean([mutual_info_score(X[:, feature], X[:, f]) for f in selected_features])
            score = relevance[feature] - redundancy

            # 최대 점수를 가진 특징 선택
            if score > max_score:
                max_score = score
                best_feature = feature

        # 선택된 특징 update
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
    return selected_features

# 예시 데이터
X = np.array([[1, 2, 3, 4],
              [2, 3, 4, 5],
              [3, 4, 5, 6],
              [4, 5, 6, 7],
              [5, 6, 7, 8]])

y = np.array([0, 1, 0, 1, 0])

# mRMR 알고리즘을 사용하여 2개의 특징 선택
selected_features = mRMR(X, y, num_features=2)
print(f"선택된 특징 인덱스: {selected_features}")
