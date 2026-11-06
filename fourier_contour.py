"""
이미지(또는 직접 그린 그림)의 윤곽선을 따서 푸리에 급수로 근사하기

사용법
  python fourier_contour.py 그림.png                  # 이미지 윤곽선 -> N별 비교 그림
  python fourier_contour.py 그림.png --Ns 3 10 30 80  # 비교할 N 직접 지정
  python fourier_contour.py 그림.png --gif            # 회전하는 원이 그림을 그려 나가는 GIF도 생성
  python fourier_contour.py --draw                    # 마우스로 직접 그리기 (GUI 필요)
  python fourier_contour.py --demo                    # 이미지 없이 예시 도형으로 테스트
  python fourier_contour.py 그림.png --tol 0.005      # 허용 오차를 바꿔서 N 자동 선택
  python fourier_contour.py 그림.png --errplot        # N에 따른 오차 그래프도 저장
  python fourier_contour.py 그림.png --live           # 창에서 N을 늘려 가며 그리는 과정 보기 (슬라이더, ↑↓ 키)
  python fourier_contour.py 그림.png --grow           # 같은 과정을 GIF로 저장 (그림_grow.gif)
  python fourier_contour.py --serve                   # 태블릿에서 그린 그림을 와이파이로 바로 받아서 그리기

태블릿 연결 (--serve)
  - 노트북과 태블릿을 같은 와이파이에 연결한다. 학교 와이파이가 기기끼리 연결을 막으면
    노트북의 모바일 핫스팟을 켜고 태블릿을 거기에 연결한다.
  - 실행하면 나오는 주소(예: http://192.168.0.12:8000)를 태블릿 브라우저에 입력하면 그림판이 열린다.
  - 참여자가 그리고 '보내기'를 누르면 drawings 폴더에 저장되고, 바로 원들이 그리는 창이 뜬다.
    새 그림이 오면 창이 자동으로 새 그림으로 바뀐다. 끝낼 때는 창을 닫고 cmd에서 Ctrl+C.
  - 처음 실행할 때 윈도우 방화벽 창이 뜨면 개인·공용 네트워크를 모두 체크하고 '액세스 허용'을 누른다.

결과 파일 (그림.png를 넣었을 때, 명령을 실행한 폴더에 저장)
  그림_compare.png  N별 비교 그림        그림_coef.csv     k별 a_k, b_k, c_k, d_k (엑셀로 열림)
  그림_error.png    오차 그래프(--errplot) 그림_fourier.gif 회전하는 원 애니메이션(--gif)

팁
  - 흰 종이에 검은 펜으로 진하게 그린 그림이 가장 잘 됩니다.
  - 배경이 어두운 이미지라면 --invert 옵션을 붙이세요.
  - 떨어진 선(눈, 입 등)과 안쪽 선까지 모든 윤곽선을 왕복 다리로 이어서 씁니다.
    다리 구간은 펜을 들고 지나가서 선이 남지 않습니다. 다리 선도 보려면 --bridges 를 붙이세요.
    예전처럼 가장 큰 윤곽선 하나만 쓰려면 --largest 를 붙이세요.
  - 휴대폰 사진은 조명 얼룩을 펴고, 사진 가장자리에 닿은 부분(책상·종이 테두리)을 자동으로 뺍니다.
    그림이 종이 끝에 닿아서 잘린다면 --keep-border 를 붙이세요.

오차와 N
  - 그림을 normalize(긴 쪽 반폭 = 1)한 뒤, 같은 t에서 원래 점과 근사 점 사이 거리의
    최댓값(최대 오차)이 --tol 이하가 되는 가장 작은 N을 자동으로 고릅니다.
  - 기본값 0.01은 그림 전체 크기의 약 0.5%로, 보통 펜 선 굵기의 절반 정도입니다.
  - 더 촘촘하게 그리려면 --tol 을 낮추세요 (예: --tol 0.005). N 상한은 기본 1000 (--nmax로 바꿈).
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------- 1. 윤곽선 추출
def read_image(path):
    """이미지를 cv2와 같은 BGR(투명 배경이면 BGRA) 배열로 읽는다.
    - cv2.imread는 윈도우에서 경로에 한글이 있으면(바탕 화면 등) 열지 못해서 PIL로 연다.
    - 휴대폰 사진의 회전 정보(EXIF)를 반영해 세로로 찍은 사진이 눕지 않게 한다."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        im = ImageOps.exif_transpose(Image.open(path))
    except FileNotFoundError:
        raise FileNotFoundError(f"파일을 찾을 수 없어요: {path} (파일 이름과 폴더를 확인해 주세요)")
    except UnidentifiedImageError:
        raise RuntimeError(f"이미지로 읽을 수 없는 파일이에요: {path} (jpg나 png로 저장해 주세요)")
    if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info:
        return np.array(im.convert("RGBA"))[:, :, [2, 1, 0, 3]]
    return np.ascontiguousarray(np.array(im.convert("RGB"))[:, :, ::-1])


def flatten_light(gray):
    """조명 얼룩(한쪽이 어둡거나 그림자 진 사진)을 편다.
    종이로 보이는 밝은 부분만 써서 '조명 밝기'를 부드럽게 추정하고 사진을 그 값으로 나눈다.
    -> 종이는 고르게 하얘지고, 선과 색칠한 부분은 어두운 채로 남는다."""
    import cv2

    h, w = gray.shape
    sw = 96
    sh = max(1, round(sw * h / w))
    v = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
    paper = (v >= np.percentile(v, 60)).astype(np.float32)   # 처음엔 가장 밝은 40%를 종이로 본다
    for _ in range(4):
        num = cv2.GaussianBlur(v * paper, (0, 0), 0.08 * sw)
        den = cv2.GaussianBlur(paper, (0, 0), 0.08 * sw)
        light = num / np.maximum(den, 1e-3)
        paper = (v >= 0.85 * light).astype(np.float32)        # 조명보다 15% 넘게 어두우면 종이가 아니다
    light = cv2.resize(light, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(gray / np.maximum(light, 1.0) * 255, 0, 255).astype(np.uint8)


def clear_border(binary):
    """사진 가장자리에 닿은 덩어리(책상, 종이 테두리, 가장자리 그림자)를 지운다."""
    import cv2

    _, lab = cv2.connectedComponents(binary)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    edge = edge[edge != 0]
    out = binary.copy()
    out[np.isin(lab, edge)] = 0
    return out, len(edge) > 0


def contour_from_image(path, invert=False, blur=3, largest=False, keep_border=False, max_side=1600, min_frac=0.02):
    """이미지의 윤곽선을 (x, y) 점 배열과, 각 점이 다리 위에 있는지(펜을 드는지) 표시로 반환한다.
    largest=False(기본): 모든 윤곽선을 왕복 다리로 이어 끊긴 곳 없는 한 붓 경로로 만든다.
    largest=True: 가장 큰 바깥 윤곽선 하나만 쓴다(예전 방식).
    keep_border=True: 사진 가장자리에 닿은 부분도 그림으로 쓴다."""
    import cv2

    img = read_image(path)

    # 투명 배경(PNG 알파)이 있으면 알파 채널을 마스크로 사용
    use_alpha = img.shape[2] == 4 and img[:, :, 3].min() < 255
    if use_alpha:
        gray = np.ascontiguousarray(img[:, :, 3])
    else:
        gray = cv2.cvtColor(np.ascontiguousarray(img[:, :, :3]), cv2.COLOR_BGR2GRAY)
        if invert:
            gray = 255 - gray

    # 휴대폰 원본처럼 큰 사진은 긴 변을 max_side로 줄인다 (속도와 잡음 기준을 맞추기 위해)
    s = max_side / max(gray.shape)
    if s < 1:
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    if not use_alpha:
        gray = flatten_light(gray)

    if blur:
        gray = cv2.GaussianBlur(gray, (2 * blur + 1, 2 * blur + 1), 0)

    # Otsu로 자동 이진화 (어두운 도형 = 전경)
    if use_alpha:
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 사인펜으로 칠하다 남은 작은 빈틈과 살짝 끊긴 선을 메운다
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # RETR_EXTERNAL은 바깥 윤곽선만 줘서 얼굴 안의 눈·입 같은 선이 빠진다 -> RETR_LIST로 전부 받는다
    mode = cv2.RETR_EXTERNAL if largest else cv2.RETR_LIST
    min_len = min_frac * max(binary.shape)   # 둘레가 이미지 긴 변의 2%(기본) 미만인 점·얼룩은 버린다

    def find(b):
        cnts, _ = cv2.findContours(b, mode, cv2.CHAIN_APPROX_NONE)
        return [c for c in cnts if cv2.arcLength(c, True) >= min_len]

    cnts, removed = find(binary), False
    if not keep_border:
        cleared, removed = clear_border(binary)
        if removed and find(cleared):      # 지우고 나서 아무것도 안 남으면 그림이 가장자리에 닿은 것 -> 지우지 않는다
            cnts = find(cleared)
            print("참고: 사진 가장자리에 닿은 부분(책상·종이 테두리 등)은 뺐어요. 그림이 잘렸다면 --keep-border")
    if not cnts:
        raise RuntimeError("윤곽선을 찾지 못했어요. 펜 선이 너무 연하거나, 어두운 종이라면 --invert 를 붙여 보세요.")
    if largest:
        c = max(cnts, key=cv2.contourArea)[:, 0, :].astype(float)  # (n, 2)
        return c, np.zeros(len(c), bool)
    return join_contours([c[:, 0, :].astype(float) for c in cnts])


def join_contours(cnts):
    """여러 윤곽선을 '갔다가 같은 길로 돌아오는 다리'로 이어 한 붓 닫힌 경로로 만든다.
    - 가장 긴 윤곽선에서 시작해, 가장 가까운 윤곽선끼리 차례로 잇는다(최소 신장 트리, Prim).
    - 다리를 왕복하므로 경로에 끊긴 곳(점프)이 없다 -> 깁스 현상(끊긴 곳에서 9% 튀어나옴)이 생기지 않는다.
    - 반환: (경로 점들, up) — up[p]가 True면 p-1번 점에서 p번 점으로 가는 구간이 다리다(펜을 든다)."""
    from scipy.spatial import cKDTree

    C = len(cnts)
    if C == 1:
        return cnts[0], np.zeros(len(cnts[0]), bool)
    trees = [cKDTree(c) for c in cnts]

    # 윤곽선 쌍마다 가장 가까운 두 점과 그 거리
    D = np.full((C, C), np.inf)
    link = {}
    for i in range(C):
        for j in range(i + 1, C):
            d, idx = trees[i].query(cnts[j])
            k = int(np.argmin(d))
            D[i, j] = D[j, i] = d[k]
            link[(i, j)] = (int(idx[k]), k)     # (i 쪽 점 번호, j 쪽 점 번호)
            link[(j, i)] = (k, int(idx[k]))

    # Prim: 가장 긴 윤곽선(root)에서 시작해 트리를 키운다
    root = int(np.argmax([len(c) for c in cnts]))
    children = {i: [] for i in range(C)}
    in_tree = np.zeros(C, bool)
    in_tree[root] = True
    best, parent = D[root].copy(), np.full(C, root)
    for _ in range(C - 1):
        j = int(np.argmin(np.where(in_tree, np.inf, best)))
        i = int(parent[j])
        ii, jj = link[(i, j)]
        children[i].append((ii, j, jj))
        in_tree[j] = True
        upd = D[j] < best
        best, parent = np.where(upd, D[j], best), np.where(upd, j, parent)

    # 트리를 따라 한 바퀴: 윤곽선을 돌다가 다리 지점에 오면 자식 윤곽선을 한 바퀴 돌고 돌아온다
    def walk(cid, start):
        c = cnts[cid]
        n = len(c)
        kids = sorted(children[cid], key=lambda a: (a[0] - start) % n)
        path, up, q = [], [], 0
        for step in range(n):
            i = (start + step) % n
            path.append(c[i]); up.append(False)
            while q < len(kids) and kids[q][0] == i:
                _, kid, kidx = kids[q]
                kp, ku = walk(kid, kidx)
                ku[0] = True                            # 다리를 건너 자식 윤곽선으로 (펜을 든다)
                path += kp; up += ku
                path.append(c[i]); up.append(True)      # 같은 다리로 되돌아옴 (펜을 든다)
                q += 1
        path.append(c[start]); up.append(False)         # 출발점으로 닫기
        return path, up

    path, up = walk(root, 0)
    return np.array(path), np.array(up)


def contour_from_drawing():
    """창에서 마우스로 점을 찍어 곡선을 만든다. (왼쪽 클릭: 점 추가, 엔터: 완료)"""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_title("클릭해서 윤곽선 점 찍기 (순서대로) · 엔터로 완료")
    pts = plt.ginput(n=-1, timeout=0, show_clicks=True)
    plt.close(fig)
    if len(pts) < 4:
        raise RuntimeError("점이 너무 적어요 (최소 4개).")
    return np.array(pts, dtype=float)


def demo_contour():
    """테스트용 예시 도형: 꽃잎 모양."""
    t = np.linspace(0, 2 * np.pi, 2000, endpoint=False)
    r = 1 + 0.35 * np.cos(6 * t) + 0.12 * np.cos(11 * t + 0.6)
    return np.column_stack([r * np.cos(t), r * np.sin(t)])


# ---------------------------------------------------------------- 2. 푸리에 근사
def resample_closed(pts, m=2048):
    """닫힌 곡선을 호 길이 기준으로 m개의 점으로 균일하게 다시 뽑는다."""
    pts = np.vstack([pts, pts[0]])                       # 닫기
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    u = np.linspace(0, s[-1], m, endpoint=False)
    return np.interp(u, s, pts[:, 0]), np.interp(u, s, pts[:, 1])


def bridge_mask(pts, up, m=2048):
    """resample_closed로 다시 찍은 점 m개 중 다리 위에 있는 점(펜을 들고 지나가는 점)을 True로 표시한다."""
    P = np.vstack([pts, pts[0]])
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    u = np.linspace(0, s[-1], m, endpoint=False)
    idx = np.searchsorted(s, u, side="right") - 1         # 각 점이 몇 번째 구간(P[idx] -> P[idx+1]) 위에 있는지
    seg_up = np.r_[np.asarray(up, bool)[1:], False]        # 구간 idx가 다리인지 (마지막 닫는 구간은 다리 아님)
    return seg_up[np.clip(idx, 0, len(seg_up) - 1)]


def pen_lift(v, up):
    """펜을 든 구간을 NaN으로 바꾼다. matplotlib은 NaN에서 선을 끊어 그리므로 그 구간은 선이 안 남는다."""
    return v if up is None else np.where(up, np.nan, v)


def normalize(x, y):
    """중심을 원점으로, 크기를 1 정도로, 위아래를 뒤집어(이미지 좌표 -> 수학 좌표)."""
    x = x - x.mean()
    y = -(y - y.mean())
    scale = max(np.abs(x).max(), np.abs(y).max())
    return x / scale, y / scale


def fourier_approx(signal, N):
    """k = 0..N 항만 남겨서 복원 (x와 y를 각각 따로 처리)."""
    M = len(signal)
    F = np.fft.rfft(signal)
    G = np.zeros_like(F)
    G[: N + 1] = F[: N + 1]
    return np.fft.irfft(G, n=M)


def circle_params(signal, N):
    """각 k에 대해 (반지름, 시작각)을 돌려준다. 회전하는 원 애니메이션용."""
    M = len(signal)
    F = np.fft.rfft(signal) / M
    amp = 2 * np.abs(F[: N + 1])
    amp[0] = np.abs(F[0])
    phase = np.angle(F[: N + 1])
    return amp, phase


# ---------------------------------------------------------------- 2-1. 오차와 N 자동 선택
def max_error(x, y, N):
    """최대 오차: 같은 t에서 원래 점과 근사 점 사이 거리의 최댓값 (normalize 후 단위)."""
    return np.hypot(x - fourier_approx(x, N), y - fourier_approx(y, N)).max()


def rms_error_curve(x, y, n_max):
    """N = 0..n_max 각각의 평균 오차(RMS).
    파르세발 정리: 평균제곱오차 = 버린 항들의 (a_k² + b_k² + c_k² + d_k²)/2 의 합
    -> 그림을 다시 그리지 않고 계수만으로 계산되고, N이 커지면 항상 줄어든다."""
    M = len(x)
    Fx, Fy = np.fft.rfft(x) / M, np.fft.rfft(y) / M
    p = 2 * (np.abs(Fx) ** 2 + np.abs(Fy) ** 2)   # k번째 항의 몫 = (a_k² + b_k² + c_k² + d_k²)/2
    p[0] = 0
    tail = np.cumsum(p[::-1])[::-1]                # tail[k] = k번째 항부터 끝까지의 합
    return np.sqrt(np.append(tail[1:], 0.0))[: n_max + 1]


def save_coefficients(x, y, N, out):
    """k = 1..N의 a_k, b_k, c_k, d_k를 CSV로 저장한다 (엑셀로 열린다).
    x(t) = Σ (a_k cos kt + b_k sin kt),  y(t) = Σ (c_k cos kt + d_k sin kt)
    그림 중심을 원점으로 옮겼으므로 상수항은 없다.
    점 M개, t_j = 2πj/M 일 때  a_k = (2/M) Σ x_j cos(k t_j),  b_k = (2/M) Σ x_j sin(k t_j)
    (c_k, d_k는 x 대신 y로 같은 계산). FFT 값 F_k로는 a_k = 2·Re F_k / M, b_k = -2·Im F_k / M.
    r_x, r_y는 k번째 원의 반지름 √(a_k² + b_k²), √(c_k² + d_k²)."""
    M = len(x)
    Fx = np.fft.rfft(x)[1: N + 1] * 2 / M
    Fy = np.fft.rfft(y)[1: N + 1] * 2 / M
    table = np.column_stack([np.arange(1, N + 1), Fx.real, -Fx.imag, Fy.real, -Fy.imag, np.abs(Fx), np.abs(Fy)])
    np.savetxt(out, table, delimiter=",", header="k,a_k,b_k,c_k,d_k,r_x,r_y", comments="",
               fmt=["%d"] + ["%.6f"] * 6)
    print("saved:", out)


def choose_N(x, y, tol=0.01, n_max=1000):
    """최대 오차가 tol 이하가 되는 가장 작은 N을 찾는다. n_max까지 못 찾으면 n_max를 돌려준다."""
    for N in range(1, n_max + 1):
        e = max_error(x, y, N)
        if e <= tol:
            return N, e
    return n_max, e


def plot_error(x, y, N_sel, tol, n_max, out="fourier_error.png"):
    """N에 따른 최대 오차·평균 오차 그래프 (로그 눈금). 허용 오차 선과 고른 N을 함께 표시."""
    n_plot = min(n_max, max(2 * N_sel, 50))                 # 고른 N의 2배까지만 그린다
    Ns = np.arange(1, n_plot + 1)
    mx = [max_error(x, y, n) for n in Ns]
    rms = rms_error_curve(x, y, n_plot)[1:]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.semilogy(Ns, mx, label="max error", color="tab:red")
    ax.semilogy(Ns, rms, label="RMS error", color="tab:blue")
    ax.axhline(tol, color="gray", ls="--", lw=1, label=f"tolerance = {tol}")
    ax.axvline(N_sel, color="tab:red", ls=":", lw=1)
    ax.text(N_sel, ax.get_ylim()[1], f" N = {N_sel}", va="top", color="tab:red")
    ax.set_xlabel("N (circles per chain)")
    ax.set_ylabel("error (half-width = 1)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print("saved:", out)


# ---------------------------------------------------------------- 3. 그리기
def plot_compare(x, y, Ns, out="fourier_contour.png", up=None):
    fig, axes = plt.subplots(1, len(Ns), figsize=(4 * len(Ns), 4.4))
    axes = np.atleast_1d(axes)
    upc = None if up is None else np.append(up, up[0])        # 다리 구간은 선을 그리지 않는다
    for ax, N in zip(axes, Ns):
        xn, yn = fourier_approx(x, N), fourier_approx(y, N)
        ax.plot(pen_lift(np.append(x, x[0]), upc), pen_lift(np.append(y, y[0]), upc), color="lightgray", lw=3)
        ax.plot(pen_lift(np.append(xn, xn[0]), upc), pen_lift(np.append(yn, yn[0]), upc), color="tab:red", lw=2)
        ax.set_title(f"N = {N}", fontsize=14)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle("Fourier approximation of the contour", fontsize=15)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print("saved:", out)


def chain_points(amp, ph, t, rotate=False):
    """원 사슬의 점들: [기준점(0,0), 상수항 뒤, 1번 원 뒤, ..., N번 원 뒤(끝점)].
    k번 원의 중심은 k번째 점이다. 원을 2D로 이어 붙여야 사슬이 된다.
    rotate=True면 각 원을 90° 돌려 배치해서, 끝점의 x가 아니라 y좌표가 신호가 된다."""
    k = np.arange(len(amp))
    vx, vy = amp * np.cos(k * t + ph), amp * np.sin(k * t + ph)
    vx[0], vy[0] = amp[0] * np.cos(ph[0]), 0.0      # k = 0: 회전하지 않는 상수항(평균)
    if rotate:
        vx, vy = -vy, vx
    P = np.zeros((len(amp) + 1, 2))
    P[1:, 0], P[1:, 1] = np.cumsum(vx), np.cumsum(vy)
    return P


def make_gif(x, y, N, out="fourier_contour.gif", frames=120, up=None):
    """x(t), y(t)를 만드는 원들이 돌면서 윤곽선을 그려 나가는 GIF."""
    import imageio.v2 as imageio
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    M = len(x)
    ax_amp, ax_ph = circle_params(x, N)
    ay_amp, ay_ph = circle_params(y, N)

    xn, yn = fourier_approx(x, N), fourier_approx(y, N)
    xs, ys = pen_lift(xn, up), pen_lift(yn, up)            # 궤적: 다리 구간은 선을 남기지 않는다
    idxs = np.linspace(0, M, frames, endpoint=False).astype(int)
    ts = 2 * np.pi * idxs / M

    # x 사슬(청록): 끝점의 x좌표 = x(t) / y 사슬(노랑): 원을 90° 돌려 배치 -> 끝점의 y좌표 = y(t)
    Px = np.array([chain_points(ax_amp, ax_ph, t) for t in ts])                 # (frames, N+2, 2)
    Py = np.array([chain_points(ay_amp, ay_ph, t, rotate=True) for t in ts])
    rx, ry = ax_amp[1:], ay_amp[1:]
    # 사슬이 그림(-1~1)과 겹치지 않게 x 사슬은 위로, y 사슬은 왼쪽으로 옮긴다
    Px[..., 1] += 1.15 - min((Px[:, 1:-1, 1] - rx).min(), Px[:, -1, 1].min())
    Py[..., 0] += -1.15 - max((Py[:, 1:-1, 0] + ry).max(), Py[:, -1, 0].max())
    xmin = min(-1.1, (Py[:, 1:-1, 0] - ry).min(), Px[..., 0].min()) - 0.05
    xmax = max(1.1, (Px[:, 1:-1, 0] + rx).max()) + 0.05
    ymin = min(-1.1, (Py[:, 1:-1, 1] - ry).min()) - 0.3            # 아래쪽은 N 글씨 자리
    ymax = max(1.1, (Px[:, 1:-1, 1] + rx).max(), Py[..., 1].max()) + 0.05
    W, H = xmax - xmin, ymax - ymin

    images = []
    for f, i in enumerate(idxs):
        fig = plt.figure(figsize=(6 * W / H, 6), dpi=80)
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal"); ax.axis("off")
        fig.patch.set_facecolor("#111")

        # 지금까지 그려진 궤적
        ax.plot(xs[: i + 1], ys[: i + 1], color="#ff8a5c", lw=2)

        for P, amp, col in [(Px[f], ax_amp, "#8fc9b8"), (Py[f], ay_amp, "#e6c84f")]:
            ax.plot(P[1:, 0], P[1:, 1], color=col, lw=0.6, alpha=0.8)          # 원을 잇는 팔
            for k in range(1, N + 1):
                if amp[k] >= 0.01:          # 아주 작은 원은 그리지만 않는다 (위치 계산에는 이미 포함)
                    ax.add_patch(plt.Circle(P[k], amp[k], fill=False, ec=col, lw=0.5, alpha=0.5))
            ax.plot([P[-1, 0], xn[i]], [P[-1, 1], yn[i]], color=col, lw=0.7, ls="--")  # 사슬 끝점 -> 그림 점

        ax.plot([xn[i]], [yn[i]], "o", color="white", ms=4)
        ax.text(0.0, ymin + 0.12, f"N = {N}", color="#ff8a5c", ha="center", fontsize=14)

        canvas.draw()
        images.append(np.asarray(canvas.buffer_rgba())[:, :, :3].copy())
        plt.close(fig)

    imageio.mimsave(out, images, duration=0.05, loop=0)
    print("saved:", out)


# ---------------------------------------------------------------- 3-1. N을 늘려 가며 그리는 과정 보기
def grow_sequence(N_auto, steps=(1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 700)):
    """원 개수를 늘려 갈 순서: steps 중 N_auto보다 작은 것 + 마지막에 N_auto."""
    return [n for n in steps if n < N_auto] + [N_auto]


class FourierView:
    """원 사슬이 그림을 그리는 화면 (x 사슬은 위, y 사슬은 왼쪽, 그림은 가운데, 아래에 N).
    show(N, i): 사슬마다 원 N개로, 한 바퀴를 frames칸으로 나눈 것 중 i칸까지 그린 모습을 그린다."""

    BG, ORANGE, TEAL, YELLOW, GHOST = "#111111", "#ff8a5c", "#8fc9b8", "#e6c84f", "#4a4338"

    def __init__(self, x, y, N_max, frames=150, controls=False, height=7.0, dpi=100, up=None):
        self.x, self.y, self.N_max, self.frames = x, y, max(2, N_max), frames
        self.sub = 8                                          # 한 칸 사이를 8등분해서 선을 매끄럽게
        self.ts = 2 * np.pi * np.arange(frames * self.sub) / (frames * self.sub)
        # 펜을 드는 구간(다리)을 그리기용 촘촘한 t에 맞춰 옮겨 둔다
        P = frames * self.sub
        self.up = None if up is None else np.asarray(up, bool)[np.arange(P) * len(x) // P]
        self.axa, self.axp = circle_params(x, self.N_max)    # x(t)를 만드는 원들의 반지름·시작각
        self.aya, self.ayp = circle_params(y, self.N_max)
        self.curves, self.errs = {}, {}
        th = np.linspace(0, 2 * np.pi, 33)
        self.ucos, self.usin = np.r_[np.cos(th), np.nan], np.r_[np.sin(th), np.nan]

        # 화면 배치: 원을 가장 많이 쓸 때의 사슬이 그림과 겹치지 않도록 사슬 위치와 화면 범위를 정한다.
        # (원 N개 사슬은 원 N_max개 사슬의 앞부분이라, N_max 사슬만 보면 모든 N이 화면에 들어온다)
        cs = [self.curve(n) for n in grow_sequence(self.N_max)]      # 원래 그림과 N별 곡선이 차지하는 범위
        gx = np.concatenate([x] + [c[0] for c in cs])
        gy = np.concatenate([y] + [c[1] for c in cs])
        left, right, bottom, top = gx.min(), gx.max(), gy.min(), gy.max()
        samp = np.linspace(0, 2 * np.pi, 90, endpoint=False)
        Px = np.array([chain_points(self.axa, self.axp, t) for t in samp])
        Py = np.array([chain_points(self.aya, self.ayp, t, rotate=True) for t in samp])
        rad = np.r_[0.0, self.axa[1:], 0.0]
        rady = np.r_[0.0, self.aya[1:], 0.0]
        self.dy = top + 0.12 - (Px[..., 1] - rad).min()        # x 사슬을 그림 위로
        self.dx = left - 0.12 - (Py[..., 0] + rady).max()      # y 사슬을 그림 왼쪽으로
        xmin = min(left, (Py[..., 0] - rady).min() + self.dx, (Px[..., 0] - rad).min()) - 0.05
        xmax = max(right, (Px[..., 0] + rad).max()) + 0.05
        ybot = min(bottom, (Py[..., 1] - rady).min())
        ymax = max(top, (Px[..., 1] + rad).max() + self.dy, (Py[..., 1] + rady).max()) + 0.05
        ymin = ybot - 0.45                                     # 아래쪽은 N 글씨 자리
        W, H = xmax - xmin, ymax - ymin
        xc = (left + right) / 2

        if controls:                                           # 창으로 띄울 때
            self.fig = plt.figure(figsize=(height * W / H, height + 0.6), dpi=dpi)
            self.ax = self.fig.add_axes([0.01, 0.09, 0.98, 0.9])
        else:                                                  # GIF로 저장할 때 (창 없이 그리기)
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            self.fig = Figure(figsize=(height * W / H, height), dpi=dpi)
            FigureCanvasAgg(self.fig)
            self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.fig.patch.set_facecolor(self.BG)
        ax = self.ax
        ax.set_facecolor(self.BG)
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal"); ax.axis("off")

        step = max(1, len(x) // 2000)                          # 원래 그림(흐리게, 다리 구간은 빼고)
        gup = None if up is None else np.r_[np.asarray(up, bool)[::step], up[0]]
        ax.plot(pen_lift(np.r_[x[::step], x[0]], gup), pen_lift(np.r_[y[::step], y[0]], gup),
                color=self.GHOST, lw=5, solid_capstyle="round")
        self.trace, = ax.plot([], [], color=self.ORANGE, lw=2)
        self.circ_x, = ax.plot([], [], color=self.TEAL, lw=0.7, alpha=0.55)
        self.arm_x, = ax.plot([], [], color=self.TEAL, lw=0.9)
        self.circ_y, = ax.plot([], [], color=self.YELLOW, lw=0.7, alpha=0.55)
        self.arm_y, = ax.plot([], [], color=self.YELLOW, lw=0.9)
        self.dash_x, = ax.plot([], [], color=self.TEAL, lw=0.8, ls="--")
        self.dash_y, = ax.plot([], [], color=self.YELLOW, lw=0.8, ls="--")
        self.pen, = ax.plot([], [], "o", color="white", ms=4)
        self.label = ax.text(xc, ybot - 0.2, "", color=self.ORANGE, ha="center", va="center", fontsize=26)
        self.err = ax.text(xc, ybot - 0.38, "", color="#9a9a9a", ha="center", va="center", fontsize=10)

    def curve(self, N):
        """원 N개로 그린 곡선 전체 (한 번 계산하면 저장해 둔다)."""
        if N not in self.curves:
            k = np.arange(1, N + 1)
            ang = np.outer(self.ts, k)
            cx = self.axa[0] * np.cos(self.axp[0]) + np.cos(ang + self.axp[1:N + 1]) @ self.axa[1:N + 1]
            cy = self.aya[0] * np.cos(self.ayp[0]) + np.cos(ang + self.ayp[1:N + 1]) @ self.aya[1:N + 1]
            self.curves[N] = (cx, cy)
        return self.curves[N]

    def _circles(self, P, amp, N):
        r = amp[1:N + 1]
        keep = r >= 0.004                                      # 안 보일 만큼 작은 원은 그리지 않는다
        c, r = P[1:N + 1][keep], r[keep]
        return ((c[:, :1] + r[:, None] * self.ucos).ravel(), (c[:, 1:] + r[:, None] * self.usin).ravel())

    def show(self, N, i):
        i = min(i, self.frames)
        t = 2 * np.pi * i / self.frames
        cx, cy = self.curve(N)
        cx, cy = pen_lift(cx, self.up), pen_lift(cy, self.up)  # 다리 구간은 선을 남기지 않는다
        if i >= self.frames:                                   # 한 바퀴 다 그림
            self.trace.set_data(np.r_[cx, cx[0]], np.r_[cy, cy[0]])
        else:
            self.trace.set_data(cx[: i * self.sub + 1], cy[: i * self.sub + 1])
        Px = chain_points(self.axa[: N + 1], self.axp[: N + 1], t); Px[:, 1] += self.dy
        Py = chain_points(self.aya[: N + 1], self.ayp[: N + 1], t, rotate=True); Py[:, 0] += self.dx
        px, py = Px[-1, 0], Py[-1, 1]                          # x 사슬 끝의 x좌표, y 사슬 끝의 y좌표 = 펜 위치
        self.arm_x.set_data(Px[1:, 0], Px[1:, 1]); self.circ_x.set_data(*self._circles(Px, self.axa, N))
        self.arm_y.set_data(Py[1:, 0], Py[1:, 1]); self.circ_y.set_data(*self._circles(Py, self.aya, N))
        self.dash_x.set_data([px, px], [Px[-1, 1], py])
        self.dash_y.set_data([Py[-1, 0], px], [py, py])
        self.pen.set_data([px], [py])
        lifted = self.up is not None and i < self.frames and self.up[i * self.sub]
        self.pen.set_color("#777777" if lifted else "white")   # 펜을 든 동안은 회색 점
        if N not in self.errs:
            self.errs[N] = max_error(self.x, self.y, N)
        self.label.set_text(f"N = {N}")
        self.err.set_text(f"max error = {self.errs[N]:.3f}")

    def frame_image(self):
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3].copy()


def make_grow_gif(x, y, N_auto, out="fourier_grow.gif", frames=48, hold=8, up=None):
    """N이 1, 2, 3, 5, 10, 20, 50, ...으로 커지면서 원들이 그림을 그리는 GIF."""
    import imageio.v2 as imageio

    seq = grow_sequence(N_auto, steps=(1, 2, 3, 5, 10, 20, 50, 100, 200, 400))
    view = FourierView(x, y, N_auto, frames=frames, height=6, dpi=80, up=up)
    images = []
    for N in seq:
        for i in range(frames + 1 + (hold if N != seq[-1] else 3 * hold)):
            view.show(N, i)
            images.append(view.frame_image())
    imageio.mimsave(out, images, duration=0.05, loop=0)
    print("saved:", out)


class LiveViewer:
    """창에서 N을 늘려 가며 그리는 과정을 보여 준다.
    자동: N이 1, 2, 3, 4, 5, 7, 10, ...으로 커지며 한 바퀴씩 그린다 (끝나면 처음부터 다시).
    직접: 슬라이더를 끌거나 ↑/↓ 키로 N을 바꾼다. space 일시정지, a 자동/직접 전환, f 전체 화면."""

    def __init__(self, x, y, N_auto, frames=150, hold=30, up=None, stop_when=None):
        from matplotlib.widgets import Slider

        self.stop_when = stop_when                                # 참이 되면 창을 닫는다 (--serve에서 새 그림이 왔을 때)
        self.view = FourierView(x, y, N_auto, frames=frames, controls=True, height=6.0, up=up)
        self.seq, self.si = grow_sequence(N_auto), 0
        self.N, self.i, self.hold = self.seq[0], 0, hold
        self.auto, self.paused = True, False
        fig = self.view.fig
        try:
            fig.canvas.manager.set_window_title("Fourier drawing")
        except AttributeError:
            pass
        sax = fig.add_axes([0.1, 0.03, 0.68, 0.03], facecolor="#333333")
        self.slider = Slider(sax, "N", 1, self.view.N_max, valinit=self.N, valstep=1, valfmt="%d",
                             color=FourierView.ORANGE)
        self.slider.label.set_color("white"); self.slider.valtext.set_color("white")
        self.slider.valtext.set_x(1.05)
        self.slider.on_changed(lambda v: self.set_N(int(v), from_slider=True))
        self.status = fig.text(0.99, 0.985, "", color="#9a9a9a", ha="right", va="top", fontsize=9)
        fig.text(0.01, 0.985, "slider / up, down: N    space: pause    a: auto    f: full screen",
                 color="#9a9a9a", ha="left", va="top", fontsize=9)
        fig.canvas.mpl_connect("key_press_event", self.on_key)

    def set_N(self, N, from_slider=False):
        self.N = int(np.clip(N, 1, self.view.N_max))
        self.auto, self.i = False, 0
        if not from_slider:
            self.slider.eventson = False
            self.slider.set_val(self.N)
            self.slider.eventson = True
        if self.paused:                                       # 멈춘 상태면 다 그린 모습을 바로 보여 준다
            self.view.show(self.N, self.view.frames)
            self._status()
            self.view.fig.canvas.draw_idle()

    def on_key(self, ev):
        if ev.key == "up":
            self.set_N(self.N + 1)
        elif ev.key == "down":
            self.set_N(self.N - 1)
        elif ev.key == " ":
            self.paused = not self.paused
            self._status(); self.view.fig.canvas.draw_idle()
        elif ev.key == "a":
            self.auto = not self.auto
            if self.auto:                                     # 자동으로 바꾸면 N = 1부터 다시
                self.si, self.N, self.i = 0, self.seq[0], 0
                self.slider.eventson = False; self.slider.set_val(self.N); self.slider.eventson = True

    def _status(self):
        self.status.set_text("paused" if self.paused else ("auto" if self.auto else "manual"))

    def tick(self, _):
        if self.stop_when is not None and self.stop_when():      # 새 그림이 들어왔으면 창을 닫고 다음 그림으로
            plt.close(self.view.fig)
            return
        self._status()
        if self.paused:
            return
        self.view.show(self.N, self.i)
        self.i += 1
        last = self.auto and self.si == len(self.seq) - 1
        if self.i > self.view.frames + (3 * self.hold if last else self.hold):
            self.i = 0
            if self.auto:                                     # 다음 N으로 (마지막이면 처음부터)
                self.si = (self.si + 1) % len(self.seq)
                self.N = self.seq[self.si]
                self.slider.eventson = False; self.slider.set_val(self.N); self.slider.eventson = True


def live_view(x, y, N_auto, up=None, stop_when=None):
    import importlib.util
    from matplotlib.animation import FuncAnimation

    if importlib.util.find_spec("tkinter") is None:          # 창을 띄우는 데 필요
        print("창을 띄울 수 없어요(tkinter 없음). 대신 --grow 로 GIF를 만들어 보세요.")
        return

    plt.rcParams["toolbar"] = "None"                          # 확대·이동 도구 막대는 숨긴다 (전시용)
    viewer = LiveViewer(x, y, N_auto, up=up, stop_when=stop_when)
    anim = FuncAnimation(viewer.view.fig, viewer.tick, interval=30, cache_frame_data=False)
    plt.show()
    return anim


# ---------------------------------------------------------------- 3-2. 태블릿에서 바로 받기 (--serve)
# 노트북이 작은 웹 서버가 되어 그림판 페이지를 보여 주고, 태블릿에서 '보내기'를 누르면 PNG를 받는다.
DRAW_PAGE = r"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<title>그림 그리기</title>
<style>
  html, body { margin: 0; height: 100%; overflow: hidden; background: #fff;
               font-family: system-ui, -apple-system, "Malgun Gothic", sans-serif; }
  #bar { position: fixed; left: 0; right: 0; top: 0; height: 64px; display: flex; align-items: center;
         gap: 10px; padding: 0 12px; box-sizing: border-box; background: #fafafa; border-bottom: 1px solid #ddd; }
  #bar .msg { flex: 1; color: #555; font-size: 18px; }
  button { font-size: 18px; padding: 10px 18px; border-radius: 10px; border: 1px solid #ccc;
           background: #fff; color: #222; touch-action: manipulation; }
  #send { background: #e8643a; border-color: #e8643a; color: #fff; font-weight: bold; }
  #pad { position: fixed; left: 0; right: 0; top: 64px; bottom: 0; width: 100%; height: calc(100% - 64px);
         touch-action: none; background: #fff; }
  #toast { position: fixed; inset: 0; display: none; align-items: center; justify-content: center;
           background: rgba(0, 0, 0, .6); color: #fff; font-size: 30px; text-align: center; padding: 24px; }
</style>
</head>
<body>
<div id="bar">
  <div class="msg">여기에 그림을 그려 주세요</div>
  <button id="undo">되돌리기</button>
  <button id="clear">모두 지우기</button>
  <button id="send">보내기</button>
</div>
<canvas id="pad"></canvas>
<div id="toast"></div>
<script>
const pad = document.getElementById('pad'), ctx = pad.getContext('2d');
const toast = document.getElementById('toast');
const PEN = 4;                                  // 펜 굵기 (화면 픽셀)
let strokes = [], cur = null, busy = false;

function style(c) { c.lineWidth = PEN; c.lineCap = 'round'; c.lineJoin = 'round'; c.strokeStyle = '#000'; c.fillStyle = '#000'; }
function drawStroke(c, s) {
  style(c);
  if (s.length === 1) { c.beginPath(); c.arc(s[0][0], s[0][1], PEN, 0, 2 * Math.PI); c.fill(); return; }
  c.beginPath(); c.moveTo(s[0][0], s[0][1]);
  for (const p of s) c.lineTo(p[0], p[1]);
  c.stroke();
}
function redraw() {
  ctx.clearRect(0, 0, pad.width, pad.height);
  for (const s of strokes) drawStroke(ctx, s);
}
function resize() {
  const r = pad.getBoundingClientRect(), d = window.devicePixelRatio || 1;
  pad.width = Math.round(r.width * d); pad.height = Math.round(r.height * d);
  ctx.setTransform(d, 0, 0, d, 0, 0);
  redraw();
}
function pos(e) { const r = pad.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }

pad.addEventListener('pointerdown', e => {
  if (busy) return;
  pad.setPointerCapture(e.pointerId);
  cur = [pos(e)]; strokes.push(cur); drawStroke(ctx, cur);
});
pad.addEventListener('pointermove', e => {
  if (!cur) return;
  const evs = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
  style(ctx); ctx.beginPath(); ctx.moveTo(cur[cur.length - 1][0], cur[cur.length - 1][1]);
  for (const ev of (evs.length ? evs : [e])) { const p = pos(ev); cur.push(p); ctx.lineTo(p[0], p[1]); }
  ctx.stroke();
});
const end = () => { cur = null; };
pad.addEventListener('pointerup', end);
pad.addEventListener('pointercancel', end);
pad.addEventListener('contextmenu', e => e.preventDefault());
window.addEventListener('resize', resize);

function show(text) { toast.textContent = text; toast.style.display = 'flex'; }
function hide() { toast.style.display = 'none'; }
document.getElementById('undo').onclick = () => { strokes.pop(); redraw(); };
document.getElementById('clear').onclick = () => { strokes = []; redraw(); };

document.getElementById('send').onclick = async () => {
  if (busy || !strokes.length) return;
  // 그림이 있는 부분만 잘라서 둘레에 흰 여백을 붙인 PNG로 만든다 (긴 변 약 1200px)
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const s of strokes) for (const [x, y] of s) {
    x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y);
  }
  const w = Math.max(x1 - x0, 1), h = Math.max(y1 - y0, 1);
  const m = 0.08 * Math.max(w, h) + PEN;
  const S = Math.min(1200 / (Math.max(w, h) + 2 * m), 3);
  const out = document.createElement('canvas');
  out.width = Math.round((w + 2 * m) * S); out.height = Math.round((h + 2 * m) * S);
  const c = out.getContext('2d');
  c.fillStyle = '#fff'; c.fillRect(0, 0, out.width, out.height);
  c.setTransform(S, 0, 0, S, (m - x0) * S, (m - y0) * S);
  for (const s of strokes) drawStroke(c, s);
  const blob = await new Promise(r => out.toBlob(r, 'image/png'));
  busy = true; show('보내는 중...');
  try {
    const res = await fetch('/upload', { method: 'POST', body: blob, headers: { 'Content-Type': 'image/png' } });
    if (!res.ok) throw new Error(res.status);
    show('보냈어요! 노트북 화면을 봐 주세요');
    strokes = []; redraw();
  } catch (err) {
    show('보내지 못했어요. 다시 눌러 주세요');
  }
  setTimeout(() => { hide(); busy = false; }, 2500);
};
resize();
</script>
</body>
</html>
"""


def local_ips():
    """노트북의 와이파이·핫스팟 IP 주소들 (태블릿 브라우저에 입력할 주소)."""
    import socket

    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))                          # 실제로 보내지는 않고, 기본 경로의 내 IP만 알아낸다
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        ips.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))


def serve(args):
    """태블릿에서 보낸 그림을 받아 drawings 폴더에 저장하고, 받는 대로 계수 계산과 원 그리기 창을 띄운다."""
    import queue
    import threading
    from datetime import datetime
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    folder = os.path.abspath("drawings")
    os.makedirs(folder, exist_ok=True)
    inbox = queue.Queue()
    page = DRAW_PAGE.encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code, body, ctype="text/plain; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):                                   # 그림판 페이지
            if self.path.split("?")[0] in ("/", "/index.html"):
                self._reply(200, page, "text/html; charset=utf-8")
            else:
                self._reply(404, b"not found")

        def do_POST(self):                                  # 그림 받기
            n = int(self.headers.get("Content-Length") or 0)
            if self.path != "/upload" or not 0 < n <= 20_000_000:
                return self._reply(400, b"bad request")
            data = self.rfile.read(n)
            if not data.startswith(b"\x89PNG"):             # PNG가 아니면 받지 않는다
                return self._reply(400, b"not a png")
            path = os.path.join(folder, datetime.now().strftime("draw_%m%d_%H%M%S_%f")[:-3] + ".png")
            with open(path, "wb") as f:
                f.write(data)
            inbox.put(path)
            self._reply(200, b"ok")

        def log_message(self, *a):                          # 접속 기록은 화면에 찍지 않는다
            pass

    try:
        server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    except OSError:
        print(f"{args.port}번 포트를 쓸 수 없어요. 다른 번호로 해 보세요: --port 8001")
        return
    threading.Thread(target=server.serve_forever, daemon=True).start()

    print("태블릿 브라우저에 아래 주소 중 하나를 입력하세요 (노트북과 같은 와이파이 또는 핫스팟):")
    for ip in local_ips() or ["(IP를 찾지 못했어요. cmd에서 ipconfig로 IPv4 주소를 확인하세요)"]:
        print(f"   http://{ip}:{args.port}")
    print(f"받은 그림과 결과는 여기에 저장됩니다: {folder}")
    print("끝내려면 창을 닫고 Ctrl+C\n그림을 기다리는 중...")

    try:
        while True:
            try:
                path = inbox.get(timeout=0.5)               # 0.5초마다 확인해야 Ctrl+C가 먹힌다
            except queue.Empty:
                continue
            print(f"\n새 그림: {os.path.basename(path)}")
            try:
                # 태블릿 그림은 배경이 깨끗하므로 가장자리 지우기는 하지 않고, 작은 점(눈 등)도 살린다
                pts, up = contour_from_image(path, keep_border=True, min_frac=0.004)
            except RuntimeError as e:
                print(e)
                continue
            x, y, pen_up = prepare(pts, up, args.bridges)
            N_auto = report_N(x, y, args.tol, args.nmax)
            base = os.path.splitext(path)[0]
            save_coefficients(x, y, N_auto, base + "_coef.csv")
            plot_compare(x, y, sorted(set(args.Ns) | {N_auto}), out=base + "_compare.png", up=pen_up)
            live_view(x, y, N_auto, up=pen_up, stop_when=lambda: not inbox.empty())
            if inbox.empty():
                print("그림을 기다리는 중...")
    except KeyboardInterrupt:
        print("\n끝냈어요.")
    finally:
        server.shutdown()


# ---------------------------------------------------------------- 4. 실행
def prepare(pts, up=None, show_bridges=False):
    """윤곽선 점들을 같은 간격의 점 M개로 다시 찍고 정규화한다. 다리 구간(펜을 드는 곳) 표시도 만든다.
    경로가 길수록 점을 많이 찍는다(약 1픽셀에 1점, 최소 2048개).
    점 간격이 허용 오차보다 충분히 작아야 오차 판정을 믿을 수 있다."""
    L = np.linalg.norm(np.diff(np.vstack([pts, pts[0]]), axis=0), axis=1).sum()
    m = max(2048, int(2 ** np.ceil(np.log2(max(L, 1.0)))))
    x, y = resample_closed(pts, m=m)
    pen_up = None if (up is None or show_bridges or not up.any()) else bridge_mask(pts, up, m)
    x, y = normalize(x, y)
    return x, y, pen_up


def report_N(x, y, tol, nmax):
    """허용 오차 안에 들어오는 가장 작은 N을 찾고 결과를 화면에 알려 준다."""
    N_auto, err = choose_N(x, y, tol=tol, n_max=nmax)
    if err <= tol:
        print(f"N = {N_auto}  (최대 오차 {err:.4f} <= {tol} / 원은 x·y 사슬 각각 {N_auto}개, 모두 {2 * N_auto}개)")
    else:
        print(f"N = {nmax}까지 늘려도 최대 오차가 {err:.4f}예요. --nmax를 늘리거나 --tol을 키워 보세요.")
    return N_auto

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", help="윤곽선을 딸 이미지 경로")
    ap.add_argument("--draw", action="store_true", help="마우스로 직접 그리기")
    ap.add_argument("--demo", action="store_true", help="예시 도형으로 테스트")
    ap.add_argument("--invert", action="store_true", help="어두운 배경/밝은 도형일 때")
    ap.add_argument("--Ns", type=int, nargs="+", default=[3, 8, 20], help="비교할 N 목록 (자동으로 찾은 N이 추가됨)")
    ap.add_argument("--tol", type=float, default=0.01, help="허용 최대 오차 (그림 긴 쪽 반폭 = 1 기준, 기본 0.01)")
    ap.add_argument("--nmax", type=int, default=1000, help="N 상한 (기본 1000)")
    ap.add_argument("--largest", action="store_true", help="가장 큰 윤곽선 하나만 쓰기 (예전 방식)")
    ap.add_argument("--bridges", action="store_true", help="떨어진 선을 잇는 다리 선도 그리기 (기본: 펜을 들고 지나감)")
    ap.add_argument("--keep-border", action="store_true", help="사진 가장자리에 닿은 부분도 그림으로 쓰기")
    ap.add_argument("--errplot", action="store_true", help="N에 따른 오차 그래프 저장")
    ap.add_argument("--gif", action="store_true", help="회전하는 원 GIF도 만들기")
    ap.add_argument("--gif-N", type=int, default=None, help="GIF에 쓸 N (기본: 자동으로 찾은 N)")
    ap.add_argument("--live", action="store_true", help="창을 띄워 N을 늘려 가며 그리는 과정 보기 (슬라이더·키보드로 조절)")
    ap.add_argument("--grow", action="store_true", help="N이 1, 2, 3, 5, 10, ...으로 커지며 그리는 GIF 만들기")
    ap.add_argument("--out", default=None, help="비교 그림 파일명 (기본: 그림이름_compare.png)")
    ap.add_argument("--serve", action="store_true", help="태블릿에서 그린 그림을 와이파이로 바로 받아서 그리기")
    ap.add_argument("--port", type=int, default=8000, help="--serve에서 쓸 포트 번호 (기본 8000)")
    args = ap.parse_args()

    if args.serve:
        serve(args)
        return

    up = None                                              # 점마다 '다리 위인지(펜을 드는지)' 표시
    if args.draw:
        pts, base = contour_from_drawing(), "draw"
    elif args.demo or not args.image:
        pts, base = demo_contour(), "demo"
    else:
        pts, up = contour_from_image(args.image, invert=args.invert, largest=args.largest,
                                     keep_border=args.keep_border)
        base = os.path.splitext(os.path.basename(args.image))[0]   # 결과 파일 이름 = 그림 파일 이름

    x, y, pen_up = prepare(pts, up, args.bridges)
    N_auto = report_N(x, y, args.tol, args.nmax)               # 허용 오차 안에 들어오는 가장 작은 N

    save_coefficients(x, y, N_auto, f"{base}_coef.csv")
    plot_compare(x, y, sorted(set(args.Ns) | {N_auto}), out=args.out or f"{base}_compare.png", up=pen_up)
    if args.errplot:
        plot_error(x, y, N_auto, args.tol, args.nmax, out=f"{base}_error.png")
    if args.gif:
        make_gif(x, y, args.gif_N or N_auto, out=f"{base}_fourier.gif", up=pen_up)
    if args.grow:
        make_grow_gif(x, y, N_auto, out=f"{base}_grow.gif", up=pen_up)
    if args.live:
        live_view(x, y, N_auto, up=pen_up)


if __name__ == "__main__":
    main()
