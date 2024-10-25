import numpy as np
from sklearn.svm import SVC
from sklearn.feature_selection import RFE

def svm_rfe(X, y, num_features):
    """
    SVM-RFE 알고리즘을 사용하여 특징을 선택하는 함수

    Parameters:
    - X: 2차원 numpy 배열 (샘플 수 x 특징 수) - 입력 특징 행렬
    - y: 1차원 numpy 배열 또는 리스트 - 대상 변수 (레이블)
    - num_features: 선택할 특징의 수

    Returns:
    - selected_features: 선택된 특징의 인덱스 리스트
    - ranking: 모든 특징에 대한 순위 (1이 가장 중요)
    """

    # SVM 모델 생성 (선형 커널을 이용하여 분류기 생성)
    svm = SVC(kernel='linear', C=1.0)

    # RFE 객체 생성
    rfe = RFE(estimator=svm, n_features_to_select=num_features, step=1)

    # RFE 알고리즘을 데이터에 적용
    rfe.fit(X, y)

    #선택된 특징의 인덱스
    selected_features = np.where(rfe.support_==True)[0]

    #모든 특징에 대한 순위 (1이 가장 중요)
    ranking = rfe.ranking_

    return selected_features, ranking

# 예시 데이터
X = np.array([[0.2, 0.8, 1.0, 0.6],
              [0.1, 0.9, 0.8, 0.5],
              [0.4, 0.7, 0.6, 0.3],
              [0.6, 0.6, 0.4, 0.2],
              [0.9, 0.5, 0.3, 0.1]])
y = np.array([0, 1, 0, 1, 0])

# SVM-RFE 알고리즘을 사용하여 2개의 특징 선택
selected_features, ranking = svm_rfe(X, y, num_features=2)

print(f"선택된 특징의 인덱스: {selected_features}")
print(f"특징 중요도 순위: {ranking}")
