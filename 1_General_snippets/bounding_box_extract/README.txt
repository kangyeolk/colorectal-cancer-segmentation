1. library �غ�

openslide�� opencv ��ġ!
scipy, numpy �ʿ�.
openslide: svs ���� ������ �ٷ絵�� ��
opencv: bounding box detection�� ����

conda install -c bioconda openslide
conda install -c conda-forge opencv 

2. �ڵ� �غ�
crop_images.py�� root directory�� �д�
crop_images.py ���� root�� �������ش�. �����η� �ξ����Ƿ� ���� �ʼ�

3. svs ������ �غ� 
crop�� svs �̹��� �����͸� ��� ��root/data �� �־�д�

4. python crop_images.py �����ϸ� �� svs �̹��� ���Ͽ� ���ؼ� ������������ crop�Ͽ�
����.tiff�� �����ϰ� ��.

5. bounding box�κ� ������ get_bounding_box���� �ϸ��.

p.s.1 ���� bounding_box_dict �� �ٿ���ڽ� ������ �ְ��ֱ������� ���� ������ �����ְ� ����.
p.s.2 ��κ��� ����Ÿ���� orig_img.read_region �Լ��κп��� ��Ƹ����� �ִµ�, �̰Ŵ� openslide�� �����ִ�
svs ���Ͽ��� �����κ��� �ҷ��鿩���� �ڵ��, bounding box region�� ���� ���Ŀ� ����ǰ� �Ǵµ�,
�� �Լ��� �ð��� ���� ��Ƹ���. 
 