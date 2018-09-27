1. library 준비

openslide와 opencv 설치!
scipy, numpy 필요.
openslide: svs 파일 형식을 다루도록 함
opencv: bounding box detection을 위함

conda install -c bioconda openslide
conda install -c conda-forge opencv 

2. 코드 준비
crop_images.py를 root directory에 둔다
crop_images.py 내의 root를 설정해준다. 절대경로로 두었으므로 수정 필수

3. svs 데이터 준비 
crop할 svs 이미지 데이터를 모두 위root/data 에 넣어둔다

4. python crop_images.py 실행하면 각 svs 이미지 파일에 대해서 세포조직별로 crop하여
숫자.tiff로 생성하게 됨.

5. bounding box부분 수정은 get_bounding_box에서 하면됨.

p.s.1 현재 bounding_box_dict 에 바운딩박스 정보를 넣고있긴하지만 따로 저장은 안해주고 있음.
p.s.2 대부분의 러닝타임은 orig_img.read_region 함수부분에서 잡아먹히고 있는데, 이거는 openslide로 열려있는
svs 파일에서 일정부분을 불러들여오는 코드로, bounding box region을 구한 이후에 실행되게 되는데,
이 함수가 시간을 많이 잡아먹음. 
 