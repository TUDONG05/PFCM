import numpy as np 
from utility import distance_cdist, division_by_zero
import time
from validity import *

import sys
sys.path.append('/home/dongtu/Desktop/DVT/pcm')
from pcm.pcm import PCM
class PFPCM(PCM):
    def __init__(self, X, n_clusters, m,n, max_iter, epsilon, seed,a,b):
        self.a=a
        self.b=b
        self.n=n 
        super().__init__(X, n_clusters, m, max_iter, epsilon, seed)
        
    # giong FCM vi co rang buoc tong
    def _ktmttv(self):
        """Khởi tạo ma trận thành viên """
        np.random.seed(self.seed)
        u =  np.random.rand(self.n_data,self.n_clusters)
        # u chia cho tổng các mức độ phụ thuộc thành phần để chuẩn hóa đảm bảo tổng các phần từ cùng 1 hàng bằng 1
        return u / division_by_zero(np.sum(u,axis=1, keepdims=True))


    # gan giong ham cap nhat dh cua PCM (10)
    def _capnhat_typicality(self):
        """ham cap nhat tham so eta(giong voi ham cap nhat ma tran thanh vien cua PCM khi b=1)"""
        d = distance_cdist(self.X, self.centroids)
        mau = (1+ (self.b*d**2/ division_by_zero(self.eta))**(1/(self.m -1)))
        return 1/ division_by_zero(mau)



    # theo cong thuc cua PFCM
    def _capnhat_tamcum(self):
        """"ham cap nhat tam cum"""
        tong = self.a*(self.u**self.m) + self.b*(self.typicality**self.n)
 
        # tu = np.sum(tong.T *self.X,axis=1,keepdims=True)
        tu = np.dot(tong.T,self.X)
        
        mau = division_by_zero(np.sum(tong.T,axis=1))
        return tu/mau[:,None]
    

    
    def fit(self):
        self.typicality, self.centroids ,self.step,self.eta= super().fit(mode=2)
        return self.typicality, self.centroids, self.step
if __name__ == '__main__':
    import time
    

    
    sys.path.append('/home/dongtu/Desktop/DVT/dataset')
    from dataset import fetch_data_from_local, TEST_CASES, LabelEncoder

    from utility import round_float, extract_labels, extract_clusters, distance_cdist, division_by_zero
    from validity import *
    

    ROUND_FLOAT = 3
    EPSILON = 1e-6
    MAX_ITER = 100
    M =2
    SEED =42
    
    
    SPLIT = '\t'
    # =======================================

    def wdvl(val: float, n: int = ROUND_FLOAT) -> str:
        return str(round_float(val, n=n))

    def write_report(alg: str, index: int, process_time: float, step: int, X: np.ndarray, V: np.ndarray, U: np.ndarray, labels_all: np.ndarray) -> str:
        labels = extract_labels(U)  # Giai mo
        kqdg = [
            alg,
            wdvl(process_time, n=2),
            str(step),
            wdvl(dunn(X, labels)),  # DI
            wdvl(davies_bouldin(X, labels)),  # DB
            wdvl(partition_coefficient(U)),  # PC
            wdvl(Xie_Benie(X, V, U)),  # XB
            wdvl(classification_entropy(U)), # CE
            wdvl(silhouette(X, labels)), #SI
            wdvl(hypervolume(U,m=2)), # FHV
            wdvl(partition_entropy(U)), # PE
            # wdvl(separation(X,U,V)), #S
            # wdvl(f1_score(labels_all,labels)) # F1
        ]
        return SPLIT.join(kqdg)
    
    


    clustering_report = []
    data_id = 53
    if data_id in TEST_CASES:
        _start_time = time.time()
        _TEST = TEST_CASES[data_id]
        _dt = fetch_data_from_local(data_id)
        if not _dt:
            print('Không thể lấy dữ liệu')
            exit()
        print("Thời gian lấy dữ liệu:", round_float(time.time() - _start_time))
        X, Y = _dt['X'], _dt['Y']
        _size = f"{_dt['data']['num_instances']} x {_dt['data']['num_features']}"
        print(f'size={_size}')
        n_clusters = _TEST['n_cluster']
        # ===============================================
        # Gán nhãn cho dữ liệu
        dlec = LabelEncoder()
        labels = dlec.fit_transform(_dt['Y'].flatten())

        # Chọn ngẫu nhiên 20% dữ liệu đã được gắn nhãn
        n_labeled = int(0.2 * len(labels))

        np.random.seed(SEED)
        labeled_indices = np.random.choice(len(labels), n_labeled, replace=False)
                                        # tổng số lượng    lượng muốn chọn   chọn ko lặp lại

        labels_all = np.full_like(labels, -1)  # Gán nhãn là -1 cho tất cả các điểm
        labels_all[labeled_indices] = labels[labeled_indices]  # Gán nhãn cho 20% dữ liệu
        
        


        titles = ['Alg', 'Time', 'Step', 'DI+', 'DB-', 'PC+', 'XB-', 'CE-','SI+','FHV-',' PE-']
        print(SPLIT.join(titles))
        

        from f1.clustering.cmeans.fcm import Dfcm
        fcm = Dfcm(n_clusters=n_clusters, m=M, epsilon=EPSILON, max_iter=MAX_ITER)
        fcm.fit(data=X, seed=SEED)
        print(write_report(alg='FCM0', index=0, process_time=fcm.process_time, step=fcm.step, X=X, V=fcm.centroids, U=fcm.membership, labels_all=labels_all))

        titles = ['Alg', 'Time', 'Step', 'DI+', 'DB-', 'PC+', 'XB-', 'CE-','SI+','FHV-','PE-']
        print(SPLIT.join(titles))

        # chay thuat toan fcm
        # ===============================================================================================================================================
        sys.path.append('/home/dongtu/Desktop/DVT/fcm')
        from fcm.fcm_np import FCM
        fcm1 = FCM(X, n_clusters=n_clusters, m=M, max_iter=MAX_ITER, epsilon=EPSILON,seed=SEED)
        u_fcm,centroids_fcm,step=fcm1.fit()
        print(write_report(alg='FCM', index=0, process_time=fcm1.time, step=fcm1.step, X=X, V=fcm1.centroids, U=fcm1.u, labels_all=labels_all))
        
        # chay thuat toan ssfcm 
        # ==============================================================================================================================================
        import sys
        sys.path.append('/home/dongtu/Desktop/DVT/SSFCM')
        from SSFCM.ssfcm import SSFCM
        ssfcm = SSFCM(X, n_clusters=n_clusters, labels=labels_all,m=M, max_iter=MAX_ITER, epsilon=EPSILON,seed=SEED)
        # ssfcm.u,ssfcm.centroids,=fcm1.u,fcm1.centroids
        u,centroids,u_ngang,step=ssfcm.fit()
        print(write_report(alg='SSFCM', index=0, process_time=ssfcm.time, step=ssfcm.step, X=X, V=ssfcm.centroids, U=ssfcm.u, labels_all=labels_all))


        # chay thuat toan pcm
        # ================================================================================================================================================
        
        pcm = PCM(X, n_clusters=n_clusters, m=M, max_iter=MAX_ITER, epsilon=EPSILON,seed=SEED)
        u_pcm,centroids_pcm,step,eta=pcm.fit(mode=2)
        print(write_report(alg='PCM', index=0, process_time=pcm.time, step=pcm.step, X=X, V=pcm.centroids, U=pcm.typicality, labels_all=labels_all))

        # chay thuat toan PFCM
        # ===============================================

        pfcm = PFPCM(X, n_clusters, M,2, MAX_ITER, EPSILON, SEED,2,0.55)
        # typicality, centroids, step = pfcm.fit()

        # Khởi tạo PFCM từ kết quả của FCM
        
        pfcm.u = u_fcm  # Dùng kết quả FCM
        pfcm.centroids = centroids_fcm  # Dùng tâm cụm từ FCM
        pfcm.typicality=pcm.typicality
        typicality, centroids, step= pfcm.fit()



        print(write_report(alg='PFCM', index=0, process_time=pfcm.time, step=pfcm.step, X=X, V=pfcm.centroids, U=pfcm.typicality, labels_all=labels_all))

        labels = extract_labels(pfcm.typicality)
        print("Unique labels:", np.unique(labels))
        print("Số lượng mẫu theo từng cụm:", np.bincount(labels))