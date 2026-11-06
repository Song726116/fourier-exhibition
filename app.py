"""모바일 전시용 푸리에 윤곽선 체험 페이지.

기존 fourier_contour.py의 이미지 처리/푸리에 근사 코드를 재사용한다.
"""
from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import streamlit as st
import streamlit.components.v1 as components

import fourier_contour as core


st.set_page_config(
    page_title="내 그림을 그리는 푸리에 원",
    page_icon="🌀",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.block-container {max-width: 790px; padding-top: 1.8rem; padding-bottom: 3rem;}
.hero {background: linear-gradient(125deg, #0b1733 0%, #203f61 100%); border-radius: 22px;
       color: white; padding: 26px 22px; margin-bottom: 20px;}
.hero h1 {margin: 0 0 8px; font-size: clamp(1.7rem, 6vw, 2.35rem); line-height: 1.22;}
.hero p {color: #d3e9f9; margin: 0; font-size: 1.03rem; line-height: 1.65;}
.step {font-weight: 750; font-size: 1.12rem; margin-top: 16px; margin-bottom: 8px;}
.smallnote {color: #66758a; font-size: 0.91rem; line-height: 1.5;}
div[data-testid="stButton"] button {border-radius: 12px; min-height: 46px;}
</style>
<div class="hero">
<h1>🌀 내 그림을 그리는 푸리에 원</h1>
<p>사진 한 장이 수많은 회전하는 원으로 변한다면?<br>
직접 찍고, 원의 개수를 바꾸고, 움직임을 확인해 보세요.</p>
</div>
""", unsafe_allow_html=True)


MAX_UPLOAD_BYTES = 12 * 1024 * 1024
SAMPLE_COUNT = 4096
MAX_N = 120


def process_photo(data: bytes, *, invert: bool, largest: bool, keep_border: bool, small_lines: bool):
    """방문객 업로드 → 원본 이미지 함수 → 동일한 푸리에 전처리.

    임시 파일은 호출 후 삭제. 사진/이름을 영구 보관하지 않는다.
    전시 응답 속도를 위해 호길이 재표본 수를 4096으로 고정한다.
    """
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("사진 용량이 12MB를 넘어요. 조금 더 작은 사진을 사용해 주세요.")
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.width * source.height > 30_000_000:
                raise ValueError("사진 해상도가 너무 높아요. 작은 사진으로 다시 시도해 주세요.")
            photo = ImageOps.exif_transpose(source)
            photo.thumbnail((1400, 1400), Image.Resampling.LANCZOS)
            photo = photo.copy()
    except (UnidentifiedImageError, OSError) as err:
        raise ValueError("이미지를 읽지 못했어요. JPG 또는 PNG 사진으로 다시 시도해 주세요.") from err

    with tempfile.TemporaryDirectory(prefix="fourier_show_") as directory:
        path = Path(directory) / "input.png"
        photo.save(path, format="PNG")
        pts, up = core.contour_from_image(
            str(path), invert=invert, largest=largest,
            keep_border=keep_border, max_side=1200,
            min_frac=0.007 if small_lines else 0.02,
        )

    x, y = core.resample_closed(pts, m=SAMPLE_COUNT)
    pen_up = None if up is None or not up.any() else core.bridge_mask(pts, up, m=SAMPLE_COUNT)
    x, y = core.normalize(x, y)
    return np.asarray(x), np.asarray(y), pen_up


def compare_png(x: np.ndarray, y: np.ndarray, up: np.ndarray | None, N: int) -> bytes:
    """화면용 원본(회색)과 푸리에 근사(주황)를 겹친 PNG."""
    xx, yy = core.fourier_approx(x, N), core.fourier_approx(y, N)
    keep = slice(None, None, max(1, len(x) // 1400))
    original = np.column_stack((x[keep], y[keep]))
    approx = np.column_stack((xx[keep], yy[keep]))
    lift = None if up is None else up[keep]

    def draw_path(ax, pts, c, lw, alpha):
        if lift is None:
            ax.plot(np.r_[pts[:, 0], pts[0, 0]], np.r_[pts[:, 1], pts[0, 1]],
                    color=c, lw=lw, alpha=alpha, solid_capstyle="round")
            return
        px = np.where(lift, np.nan, pts[:, 0])
        py = np.where(lift, np.nan, pts[:, 1])
        ax.plot(np.r_[px, px[0]], np.r_[py, py[0]], color=c, lw=lw, alpha=alpha,
                solid_capstyle="round")

    fig, ax = plt.subplots(figsize=(7, 5), dpi=125)
    fig.patch.set_facecolor("#0b1120")
    ax.set_facecolor("#0b1120")
    draw_path(ax, original, "#64748b", 4, 0.65)
    draw_path(ax, approx, "#fb923c", 2.5, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    out = io.BytesIO()
    fig.savefig(out, format="png", facecolor=fig.get_facecolor(), dpi=125)
    plt.close(fig)
    return out.getvalue()


def csv_bytes(x: np.ndarray, y: np.ndarray, N: int) -> bytes:
    """원본 코드 save_coefficients와 동일한 정의의 CSV를 메모리에서 생성."""
    M = len(x)
    fx = np.fft.rfft(x)[1:N + 1] * 2 / M
    fy = np.fft.rfft(y)[1:N + 1] * 2 / M
    table = np.column_stack((np.arange(1, N + 1), fx.real, -fx.imag, fy.real,
                             -fy.imag, np.abs(fx), np.abs(fy)))
    buf = io.StringIO()
    np.savetxt(buf, table, delimiter=",", header="k,a_k,b_k,c_k,d_k,r_x,r_y",
               comments="", fmt=["%d"] + ["%.6f"] * 6)
    return buf.getvalue().encode("utf-8-sig")


def animation_html(x: np.ndarray, y: np.ndarray, up: np.ndarray | None, N: int) -> str:
    """브라우저 자체에서 동작하는 경량 애니메이션 (서버에 GIF 생성 부담 X)."""
    M = len(x)
    fx, fy = np.fft.rfft(x) / M, np.fft.rfft(y) / M

    def params(F):
        v = F[:N+1]
        amp = 2 * np.abs(v)
        amp[0] = abs(v[0])
        return [round(float(a), 7) for a in amp], [round(float(p), 7) for p in np.angle(v)]

    ax, px = params(fx)
    ay, py = params(fy)
    ix = np.linspace(0, M - 1, 750).astype(int)
    cx, cy = core.fourier_approx(x, N), core.fourier_approx(y, N)
    data = {
        "ax": ax, "px": px, "ay": ay, "py": py,
        "x": np.round(cx[ix], 5).tolist(),
        "y": np.round(cy[ix], 5).tolist(),
        "ox": np.round(x[ix], 5).tolist(),
        "oy": np.round(y[ix], 5).tolist(),
        "up": np.zeros(len(ix), dtype=int).tolist() if up is None else up[ix].astype(int).tolist(),
        "N": N,
    }
    json_data = json.dumps(data, ensure_ascii=True, separators=(",", ":"))
    # 모든 데이터는 입력 파일명이 아닌 계산된 숫자이므로 HTML 문자열 삽입이 안전하다.
    return r'''<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
* {box-sizing: border-box}
body {background:#0b1120;color:#e2e8f0;font-family:system-ui,-apple-system,sans-serif;
 margin:0;padding:10px;border-radius:16px}
#canvas {width:100%;max-width:460px;height:auto;aspect-ratio:1/1;display:block;margin:auto;background:#0b1120;border-radius:14px}
.bar{display:flex;align-items:center;justify-content:center;gap:13px;flex-wrap:wrap;padding:5px 0 8px}
button{background:#f97316;border:0;color:#fff;font-weight:700;border-radius:12px;padding:9px 20px;font-size:15px}
label{font-size:13px;color:#cbd5e1}
input{accent-color:#fb923c;width:100px;vertical-align:middle}
.legend{display:flex;justify-content:center;gap:12px;flex-wrap:wrap;font-size:12px;color:#cbd5e1}
.sw{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}
</style></head><body>
<canvas id="canvas"></canvas>
<div class="bar"><button id="toggle">⏸ 일시정지</button>
<label>속도 <input id="speed" type="range" min="0.4" max="2.5" step="0.1" value="1"></label></div>
<div class="legend"><span><i class="sw" style="background:#f97316"></i>완성되는 그림</span>
<span><i class="sw" style="background:#38bdf8"></i>x 방향 원</span>
<span><i class="sw" style="background:#fde047"></i>y 방향 원</span></div>
<script>
const d = __DATA__;
const canvas = document.getElementById('canvas'), g=canvas.getContext('2d');
let W=350,H=350, t=0, last=0, playing=true;
const speed=document.getElementById('speed'), toggle=document.getElementById('toggle');
toggle.onclick=()=>{playing=!playing;toggle.textContent=playing?'⏸ 일시정지':'▶ 재생';};
function resize(){let w=canvas.getBoundingClientRect().width;W=w;H=w;
 let r=window.devicePixelRatio||1;canvas.width=Math.round(w*r);canvas.height=Math.round(w*r);
 g.setTransform(r,0,0,r,0,0);}
window.addEventListener('resize',resize);resize();
function pathFromSamples(xs,ys,up,limit,color,width,alpha=1){
 g.strokeStyle=color;g.lineWidth=width;g.globalAlpha=alpha;g.lineCap='round';g.lineJoin='round';
 g.beginPath();let prev=false;
 for(let j=0;j<=limit;j++){
   let ok=up[j]===0;
   let X=.68*W + .16*W*xs[j],Y=.68*H - .16*W*ys[j];
   if(ok){if(prev)g.lineTo(X,Y);else g.moveTo(X,Y);}prev=ok;
 }
 g.stroke();g.globalAlpha=1;
}
function chain(amps,phs,turn,rotate,color,ox,oy,s){
 let a=0,b=0,kmax=amps.length-1;
 let lastx=ox,lasty=oy;
 g.strokeStyle=color;g.lineWidth=1.05;g.globalAlpha=.75;
 for(let k=0;k<=kmax;k++){
   let vx,vy;
   if(k===0){vx=amps[k]*Math.cos(phs[k]);vy=0;}
   else {vx=amps[k]*Math.cos(k*turn+phs[k]);vy=amps[k]*Math.sin(k*turn+phs[k]);}
   if(rotate){let u=vx;vx=-vy;vy=u;}
   const cX=ox+s*a,cY=oy-s*b;
   a+=vx;b+=vy;
   const eX=ox+s*a,eY=oy-s*b;
   if(k>0 && amps[k]>0.008){
     g.beginPath();g.arc(cX,cY,s*amps[k],0,Math.PI*2);g.stroke();
   }
   if(k>0){g.beginPath();g.moveTo(cX,cY);g.lineTo(eX,eY);g.stroke();}
   lastx=eX;lasty=eY;
 }
 g.globalAlpha=1;
 return [lastx,lasty];
}
function draw(){
 g.clearRect(0,0,W,H);let s=.16*W;
 // 원본 윤곽선은 회색, 재현 중인 선은 주황색
 pathFromSamples(d.ox,d.oy,d.up,d.x.length-1,'#526278',3,.70);
 let ix=Math.min(d.x.length-1,Math.floor(t*(d.x.length-1)));
 pathFromSamples(d.x,d.y,d.up,ix,'#fb923c',2.2,1);
 let turn=t*Math.PI*2;
 const xChain=chain(d.ax,d.px,turn,false,'#38bdf8',.68*W,.20*H,s);
 const yChain=chain(d.ay,d.py,turn,true,'#fde047',.20*W,.68*H,s);
 const pointX=.68*W+s*d.x[ix],pointY=.68*H-s*d.y[ix];
 g.strokeStyle='#94a3b8';g.setLineDash([4,5]);g.globalAlpha=.6;
 g.beginPath();g.moveTo(xChain[0],xChain[1]);g.lineTo(pointX,pointY);
 g.moveTo(yChain[0],yChain[1]);g.lineTo(pointX,pointY);g.stroke();
 g.setLineDash([]);g.globalAlpha=1;
 g.beginPath();g.fillStyle='#fff';g.arc(pointX,pointY,3,0,Math.PI*2);g.fill();
 g.fillStyle='#cbd5e1';g.font=`${Math.max(12,W*.038)}px system-ui`;
 g.fillText('N = '+d.N,13,H-14);
}
function frame(now){
 if(!last)last=now;
 let dt=Math.min(.1,(now-last)/1000);last=now;
 if(playing)t=(t+dt*Number(speed.value)/7)%1;
 draw();requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
</script></body></html>'''.replace("__DATA__", json_data)


st.markdown('<div class="step">① 그림을 준비해 보세요</div>', unsafe_allow_html=True)
mode = st.radio("입력 방법", ["📸 사진 촬영", "🖼️ 앨범에서 선택", "🌼 예시로 체험"],
                horizontal=True, label_visibility="collapsed")

uploaded = None
if mode == "📸 사진 촬영":
    uploaded = st.camera_input("흰 종이에 그린 그림을 찍어 주세요")
elif mode == "🖼️ 앨범에서 선택":
    uploaded = st.file_uploader("JPG 또는 PNG 이미지 올리기", type=["jpg", "jpeg", "png"])
else:
    st.info("예시 꽃 도형으로 바로 체험할 수 있어요. 사진이 없어도 괜찮아요!")

with st.expander("⚙️ 사진 인식 설정 (필요할 때만 열기)"):
    invert = st.checkbox("밝은 선 / 어두운 배경", value=False)
    largest = st.checkbox("가장 큰 바깥 윤곽선만 사용", value=False)
    keep_border = st.checkbox("사진 가장자리에 닿은 선도 포함", value=False)
    small_lines = st.checkbox("작은 선(눈·입 등) 더 살리기", value=False)

if uploaded is not None:
    st.image(uploaded, caption="선택한 사진", width="stretch")

click = st.button("✨ 내 그림 변환하기" if mode != "🌼 예시로 체험" else "🌼 예시 도형으로 시작하기",
                  type="primary", use_container_width=True)
if click:
    try:
        with st.spinner("윤곽선을 찾고 푸리에 계수를 계산하고 있어요..."):
            if mode == "🌼 예시로 체험":
                pts = core.demo_contour()
                x, y = core.resample_closed(pts, m=SAMPLE_COUNT)
                x, y = core.normalize(x, y)
                up = None
            elif uploaded is None:
                raise ValueError("먼저 사진을 찍거나 선택해 주세요!")
            else:
                x, y, up = process_photo(uploaded.getvalue(), invert=invert, largest=largest,
                                          keep_border=keep_border, small_lines=small_lines)
            st.session_state.result = (x, y, up)
            st.session_state.N = 10
    except (ValueError, RuntimeError, OSError, MemoryError) as error:
        st.error(str(error))

if "result" in st.session_state:
    x, y, up = st.session_state.result
    st.markdown('<div class="step">② 원의 개수를 바꿔 보세요</div>', unsafe_allow_html=True)
    N = st.slider("원 개수 N (x와 y에 각각 N개)", 1, MAX_N, key="N")
    error = core.max_error(x, y, N)
    st.caption(f"회전하는 원: x 방향 {N}개 + y 방향 {N}개 = 총 {2*N}개 · 최대 오차 {error:.4f}")
    if N <= 5:
        st.info("원이 적으면 단순한 곡선만 보여요. N을 늘려 보세요!")
    st.image(compare_png(x, y, up, N), caption="회색: 원본 윤곽선  ·  주황색: 푸리에 근사", width="stretch")
    st.markdown('<div class="step">③ 움직이는 원을 관찰해 보세요</div>', unsafe_allow_html=True)
    st.markdown('<div class="smallnote">파란 원들이 가로 좌표를, 노란 원들이 세로 좌표를 만들어요. 두 좌표가 만나는 점이 그림을 그려요.</div>', unsafe_allow_html=True)
    components.html(animation_html(x, y, up, N), height=560, scrolling=False)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("🖼️ 결과 이미지 저장", data=compare_png(x, y, up, N),
                           file_name="fourier_result.png", mime="image/png", use_container_width=True)
    with c2:
        st.download_button("📊 계수 CSV 저장", data=csv_bytes(x, y, N),
                           file_name="fourier_coefficients.csv", mime="text/csv", use_container_width=True)

st.divider()
st.markdown("**원리를 간단히 알아볼까?**")
st.markdown("푸리에 급수는 복잡한 곡선을 여러 개의 사인·코사인 파동으로 나타내는 방법이야. "
            "원의 회전이 파동을 만들고, 그 파동들을 더하면 그림의 x좌표와 y좌표를 되살릴 수 있어.")
st.caption("💡 팁: 흰 종이에 검은 펜으로 굵고 선명하게 그려 주세요. 사람 얼굴·개인정보는 찍지 말아 주세요. 사진은 처리 중에만 임시로 사용하고, 앱 코드에서 영구 저장하지 않아요.")
