/* Hình động cho slide. Mỗi <svg class="anim" data-kind="..."> được dựng một lần,
   rồi cập nhật mỗi khung hình khi slide chứa nó đang hiện. t tính bằng giây
   kể từ lúc slide được mở, nên mỗi lần quay lại slide hình động chạy lại từ đầu. */
(function () {
  const NS = 'http://www.w3.org/2000/svg';
  const FONT = '"Be Vietnam Pro","Helvetica Neue",Arial,sans-serif';
  const C = {
    navy: '#16324f', ink: '#1f2933', muted: '#5b6672', faint: '#8a94a0', line: '#d9dee4',
    soft: '#f3f5f7', blue: '#2a7bd5', orange: '#e8643b', green: '#1ba97a', teal: '#1f9fb0',
    amber: '#e0a100', red: '#d64545', purple: '#7c5cc4', gray: '#9aa5b1', bone: '#c9d0d8',
  };

  // ---------------------------------------------------------------- tiện ích
  function el(parent, tag, attrs, text) {
    const e = document.createElementNS(NS, tag);
    if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }
  function set(e, attrs) { for (const k in attrs) e.setAttribute(k, attrs[k]); }
  function text(parent, x, y, str, o = {}) {
    return el(parent, 'text', {
      x, y, 'font-size': o.size || 14, fill: o.fill || C.muted, 'font-weight': o.w || 500,
      'text-anchor': o.anchor || 'start', 'font-family': FONT, 'dominant-baseline': o.base || 'auto',
    }, str);
  }
  const lerp = (a, b, t) => a + (b - a) * t;
  const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
  const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
  const wave = (t, P) => 0.5 - 0.5 * Math.cos((2 * Math.PI * t) / P);
  const phase = (t, a, b) => clamp((t - a) / (b - a));
  function rgb(h) { return [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)); }
  function mix(a, b, t) {
    const A = rgb(a), B = rgb(b);
    return `rgb(${A.map((v, i) => Math.round(v + (B[i] - v) * clamp(t))).join(',')})`;
  }
  const tint = (c, k) => mix(c, '#ffffff', k);
  function vb(svg, w, h) { svg.innerHTML = ''; svg.setAttribute('viewBox', `0 0 ${w} ${h}`); }
  const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  function softmax(xs) { const m = Math.max(...xs); const e = xs.map((x) => Math.exp(x - m)); const s = e.reduce((a, b) => a + b, 0); return e.map((x) => x / s); }
  function arrow(g, x1, y1, x2, y2, color = C.faint, w = 2) {
    el(g, 'line', { x1, y1, x2: x2 - 8, y2, stroke: color, 'stroke-width': w });
    el(g, 'path', { d: `M${x2 - 10} ${y2 - 6}L${x2} ${y2}L${x2 - 10} ${y2 + 6}z`, fill: color });
  }

  // ---------------------------------------------------------------- khung xương dùng chung
  // Toạ độ trong ô 400×420: thân trên, hai tay, mỗi bàn tay 5 đầu ngón.
  const BASE = [
    [200, 70], [186, 60], [214, 60], [168, 66], [232, 66], [200, 90], [200, 124],
    [146, 136], [254, 136], [120, 210], [280, 210], [112, 282], [288, 282], [168, 330], [232, 330],
    [92, 306], [100, 316], [112, 320], [124, 316], [134, 304],
    [266, 304], [276, 316], [288, 320], [300, 316], [308, 306],
  ];
  const BONES = [[0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [5, 6], [6, 7], [6, 8], [7, 9], [9, 11], [8, 10], [10, 12], [7, 13], [8, 14], [13, 14]];
  for (let i = 0; i < 5; i++) { BONES.push([11, 15 + i]); BONES.push([12, 20 + i]); }
  const WRIST = 12;

  // s = 0: tay buông; s = 1: tay phải (bên phải màn hình) đưa lên cạnh miệng.
  function pose(s) {
    const P = BASE.map((p) => p.slice());
    const wr = [lerp(288, 226, s), lerp(282, 108, s)];
    P[10] = [lerp(280, 290, s), lerp(210, 180, s)];
    P[12] = wr;
    const a = -s * 2.7, ca = Math.cos(a), sa = Math.sin(a);
    for (let i = 0; i < 5; i++) {
      const dx = BASE[20 + i][0] - 288, dy = BASE[20 + i][1] - 282;
      P[20 + i] = [wr[0] + dx * ca - dy * sa, wr[1] + dx * sa + dy * ca];
    }
    return P;
  }

  function hops(src) {
    const adj = BASE.map(() => []);
    BONES.forEach(([a, b]) => { adj[a].push(b); adj[b].push(a); });
    const d = BASE.map(() => Infinity); d[src] = 0;
    const q = [src];
    while (q.length) { const u = q.shift(); adj[u].forEach((v) => { if (d[v] === Infinity) { d[v] = d[u] + 1; q.push(v); } }); }
    return d;
  }

  function skeleton(parent, o = {}) {
    const g = el(parent, 'g');
    const s = { g, edges: [], nodes: [], P: BASE };
    s.head = el(g, 'circle', { cx: 200, cy: 70, r: 36, fill: 'none', stroke: C.bone, 'stroke-width': o.bw || 5 });
    const ge = el(g, 'g');
    s.extra = el(g, 'g');
    const gn = el(g, 'g');
    BONES.forEach(() => s.edges.push(el(ge, 'line', { stroke: C.bone, 'stroke-width': o.bw || 5, 'stroke-linecap': 'round' })));
    BASE.forEach((p, i) => s.nodes.push(el(gn, 'circle', { r: i >= 15 ? 4 : 7, fill: C.gray })));
    s.update = (P) => {
      s.P = P;
      BONES.forEach(([a, b], k) => set(s.edges[k], { x1: P[a][0], y1: P[a][1], x2: P[b][0], y2: P[b][1] }));
      P.forEach((p, i) => set(s.nodes[i], { cx: p[0], cy: p[1] }));
    };
    s.reset = () => {
      s.nodes.forEach((n, i) => set(n, { fill: C.gray, r: i >= 15 ? 4 : 7, opacity: 1 }));
      s.edges.forEach((e) => set(e, { stroke: C.bone, opacity: 1 }));
    };
    s.update(BASE);
    return s;
  }

  function title(svg, x, y) { return text(svg, x, y, '', { size: 18, w: 700, fill: C.navy }); }
  function particles(parent, n, color, r = 5) {
    const out = [];
    for (let i = 0; i < n; i++) out.push(el(parent, 'circle', { r, fill: color, opacity: 0 }));
    return out;
  }
  function divider(svg) { el(svg, 'line', { x1: 398, y1: 20, x2: 398, y2: 400, stroke: C.line, 'stroke-width': 1.5, 'stroke-dasharray': '4 6' }); }

  const KINDS = {};

  // ================================================================ tổng quan hệ thống
  KINDS.overview = {
    build(svg) {
      vb(svg, 1440, 600);
      const s = { boxes: [], segs: [], tags: [] };
      const lane = (y, h, label, color) => {
        el(svg, 'rect', { x: 6, y, width: 1428, height: h, rx: 14, fill: 'none', stroke: color, 'stroke-dasharray': '7 6', 'stroke-width': 1.5 });
        text(svg, 26, y + 28, label, { size: 15, w: 700, fill: color });
      };
      lane(10, 262, 'THỜI GIAN THỰC · trên máy người dùng', C.blue);
      lane(330, 262, 'NGOẠI TUYẾN · dữ liệu và huấn luyện', C.green);
      const on = [
        ['Webcam', '25 khung/s · 224×224', C.faint], ['MediaPipe', '76 điểm (x, y, z)', C.blue],
        ['Tách ký hiệu', 'PoseStateMLP · P > 0,3', C.amber], ['Chuẩn hoá', 'theo vai · 64 khung', C.amber],
        ['CTR-GCN', '422 từ · 31 ms', C.green], ['Ngưỡng tin cậy', '0,4 · 0,7 · 0,8', C.teal], ['Giao diện', 'chữ + 5 gợi ý', C.purple],
      ];
      const off = [
        ['VSL400', '74.259 clip · 28 người', C.faint], ['Trích keypoint', 'MediaPipe → (T, 76, 3)', C.blue],
        ['Lọc + tăng cường', '422 từ · ≥ 20 clip/từ', C.amber], ['Huấn luyện', '14 mô hình · 2 × T4', C.green],
        ['Đánh giá', 'VSL400 · QIPEDC · từ lạ', C.green], ['Chọn mô hình', 'CTR-GCN J76', C.navy],
      ];
      const segLayer = el(svg, 'g');
      const box = (x, y, w, h, [t1, t2, col], i, laneNo) => {
        const g = el(svg, 'g', { opacity: 0 });
        const r = el(g, 'rect', { x, y, width: w, height: h, rx: 10, fill: tint(col, 0.88), stroke: col, 'stroke-width': 2 });
        text(g, x + w / 2, y + h / 2 - 4, t1, { size: 19, w: 700, fill: C.ink, anchor: 'middle' });
        text(g, x + w / 2, y + h / 2 + 22, t2, { size: 13.5, fill: C.muted, anchor: 'middle' });
        s.boxes.push({ g, r, i, laneNo, col });
        return { x, y, w, h };
      };
      const seg = (x1, x2, y, laneNo) => {
        const g = el(segLayer, 'g', { opacity: 0 });
        arrow(g, x1 + 2, y, x2 - 2, y, C.faint, 2);
        const ps = particles(g, 2, laneNo ? C.green : C.blue, 5);
        s.segs.push({ g, x1, x2, y, ps, laneNo });
      };
      let prev = null;
      on.forEach((b, i) => { const r = box(20 + i * 202, 84, 178, 112, b, i, 0); if (prev) seg(prev.x + prev.w, r.x, 140, 0); prev = r; });
      prev = null;
      off.forEach((b, i) => { const r = box(20 + i * 236, 410, 200, 112, b, i, 1); if (prev) seg(prev.x + prev.w, r.x, 466, 1); prev = r; });
      const tag = (cx, label, col) => {
        const g = el(svg, 'g', { opacity: 0 });
        const w = label.length * 7.6 + 24;
        el(g, 'line', { x1: cx, y1: 196, x2: cx, y2: 214, stroke: col, 'stroke-width': 2 });
        el(g, 'rect', { x: cx - w / 2, y: 214, width: w, height: 30, rx: 15, fill: tint(col, 0.86), stroke: col });
        text(g, cx, 234, label, { size: 13.5, w: 600, fill: col, anchor: 'middle' });
        s.tags.push(g);
      };
      tag(20 + 2 * 202 + 89, '≥ 50% khung lỗi → ký lại', C.red);
      tag(20 + 5 * 202 + 89, '< 0,7 → 5 gợi ý hoặc ký lại', C.amber);
      const fromX = 20 + 5 * 236 + 100, toX = 20 + 4 * 202 + 89;
      s.deploy = el(svg, 'path', { d: `M${fromX} 410V300H${toX}V202`, fill: 'none', stroke: C.navy, 'stroke-width': 2.5, 'stroke-dasharray': '9 7', opacity: 0 });
      s.deployHead = el(svg, 'path', { d: `M${toX - 7} 212L${toX} 200L${toX + 7} 212z`, fill: C.navy, opacity: 0 });
      s.deployLabel = text(svg, (fromX + toX) / 2, 292, 'triển khai checkpoint', { size: 14, w: 600, fill: C.navy, anchor: 'middle', });
      set(s.deployLabel, { opacity: 0 });
      return s;
    },
    step(s, t) {
      s.boxes.forEach((b) => {
        const d = b.laneNo ? 2.2 + b.i * 0.22 : b.i * 0.22;
        const a = ease(phase(t, d, d + 0.45));
        set(b.g, { opacity: a, transform: `translate(0 ${(1 - a) * 14})` });
        const hot = b.laneNo === 0 && t > 4.2 && Math.floor((t - 4.2) / 0.8) % 7 === b.i;
        set(b.r, { 'stroke-width': hot ? 4.5 : 2, fill: hot ? tint(b.col, 0.72) : tint(b.col, 0.88) });
      });
      s.segs.forEach((g) => {
        const start = g.laneNo ? 3.4 : 1.6;
        set(g.g, { opacity: ease(phase(t, start, start + 0.5)) });
        g.ps.forEach((p, k) => {
          const u = (t * (g.laneNo ? 0.45 : 0.9) + k * 0.5) % 1;
          set(p, { cx: lerp(g.x1 + 4, g.x2 - 10, u), cy: g.y, opacity: t > start ? Math.sin(Math.PI * u) : 0 });
        });
      });
      s.tags.forEach((g) => set(g, { opacity: ease(phase(t, 1.8, 2.4)) }));
      const a = ease(phase(t, 3.8, 4.4));
      set(s.deploy, { opacity: a, 'stroke-dashoffset': -t * 28 });
      set(s.deployHead, { opacity: a });
      set(s.deployLabel, { opacity: a });
    },
  };

  // ================================================================ tách ký hiệu thời gian thực
  KINDS.segment = {
    build(svg) {
      vb(svg, 1440, 580);
      const s = {};
      const bx = (x, y, w, h, t1, t2, col, size = 21) => {
        const r = el(svg, 'rect', { x, y, width: w, height: h, rx: 12, fill: tint(col, 0.9), stroke: col, 'stroke-width': 2 });
        text(svg, x + w / 2, y + h / 2 + (t2 ? -4 : 6), t1, { size, w: 700, fill: C.ink, anchor: 'middle' });
        if (t2) text(svg, x + w / 2, y + h / 2 + 22, t2, { size: 14, fill: C.muted, anchor: 'middle' });
        return { r, col };
      };
      s.states = [
        bx(40, 40, 290, 104, 'NGHỈ', 'giữ 1 khung mới nhất', C.faint),
        bx(430, 40, 290, 104, 'ĐANG KÝ', 'thêm mọi khung hợp lệ', C.green),
        bx(820, 40, 290, 104, 'ĐÓNG CLIP', 'đệm 5 khung nghỉ mỗi đầu', C.amber),
      ];
      s.outs = [bx(1160, 14, 270, 70, 'Chấm điểm → hiện từ', '', C.green, 17), bx(1160, 104, 270, 70, '≥ 50% khung lỗi → ký lại', '', C.red, 17)];
      arrow(svg, 330, 92, 430, 92); text(svg, 380, 80, 'P > 0,3', { size: 14, w: 600, fill: C.muted, anchor: 'middle' });
      arrow(svg, 720, 92, 820, 92); text(svg, 770, 80, 'P ≤ 0,3', { size: 14, w: 600, fill: C.muted, anchor: 'middle' });
      el(svg, 'path', { d: 'M1110 80L1160 50', stroke: C.faint, 'stroke-width': 2, fill: 'none' });
      el(svg, 'path', { d: 'M1110 104L1160 139', stroke: C.faint, 'stroke-width': 2, fill: 'none' });
      el(svg, 'path', { d: 'M965 144C900 210 260 210 185 146', stroke: C.faint, 'stroke-width': 1.8, fill: 'none', 'stroke-dasharray': '6 5' });
      text(svg, 575, 196, 'chờ tư thế nghỉ mới rồi nhận ký hiệu tiếp theo', { size: 14, fill: C.faint, anchor: 'middle' });

      // biểu đồ P(đang ký)
      const X0 = 120, X1 = 1390, Y0 = 540, Y1 = 290;
      Object.assign(s, { X0, X1, Y0, Y1 });
      const bump = (x, c, w) => Math.exp(-Math.pow((x - c) / w, 6));
      s.p = (x) => 0.06 + 0.88 * bump(x, 0.3, 0.12) + 0.88 * bump(x, 0.72, 0.1);
      s.X = (x) => lerp(X0, X1, x);
      s.Y = (p) => lerp(Y0, Y1, p);
      // các đoạn p > 0,3
      s.ints = []; let open = null;
      for (let x = 0; x <= 1.0001; x += 0.001) {
        const on = s.p(x) > 0.3;
        if (on && open === null) open = x;
        if (!on && open !== null) { s.ints.push([open, x]); open = null; }
      }
      s.shade = s.ints.map(() => el(svg, 'rect', { y: Y1 - 10, height: Y0 - Y1 + 10, fill: tint(C.green, 0.86), width: 0 }));
      el(svg, 'line', { x1: X0, y1: Y0, x2: X1, y2: Y0, stroke: C.muted, 'stroke-width': 1.5 });
      el(svg, 'line', { x1: X0, y1: Y0, x2: X0, y2: Y1 - 10, stroke: C.muted, 'stroke-width': 1.5 });
      text(svg, X0 - 12, s.Y(1) + 5, '1', { size: 13, anchor: 'end' });
      text(svg, X0 - 12, Y0 + 5, '0', { size: 13, anchor: 'end' });
      el(svg, 'line', { x1: X0, y1: s.Y(0.3), x2: X1, y2: s.Y(0.3), stroke: C.red, 'stroke-dasharray': '7 6', 'stroke-width': 1.6 });
      text(svg, X0 - 12, s.Y(0.3) + 5, '0,3', { size: 13, fill: C.red, w: 700, anchor: 'end' });
      text(svg, X0 - 70, (Y0 + Y1) / 2, 'P(đang ký)', { size: 14, w: 600, fill: C.muted, anchor: 'middle' });
      set(svg.lastChild, { transform: `rotate(-90 ${X0 - 70} ${(Y0 + Y1) / 2})` });
      text(svg, X1, Y0 + 32, 'thời gian →', { size: 13, fill: C.faint, anchor: 'end' });
      s.curve = el(svg, 'path', { fill: 'none', stroke: C.blue, 'stroke-width': 3.5, 'stroke-linejoin': 'round' });
      s.cursor = el(svg, 'line', { y1: Y1 - 10, y2: Y0, stroke: C.navy, 'stroke-width': 1.5, 'stroke-dasharray': '3 4' });
      s.dot = el(svg, 'circle', { r: 7, fill: C.navy });
      s.queue = text(svg, X1, Y1 - 24, '', { size: 16, w: 700, fill: C.navy, anchor: 'end' });
      // khung lỗi trong ký hiệu thứ hai
      const [a2, b2] = s.ints[1];
      s.ticks = [];
      const lost = [1, 2, 4, 5, 7, 8, 9, 11, 12, 14, 15];
      for (let k = 0; k < 18; k++) {
        const x = lerp(a2, b2, (k + 0.5) / 18);
        const bad = lost.includes(k);
        s.ticks.push({ x, e: el(svg, 'rect', { x: s.X(x) - 3, y: Y0 + 8, width: 6, height: 14, rx: 2, fill: bad ? C.red : C.green, opacity: 0 }) });
      }
      s.tickLabel = text(svg, s.X((a2 + b2) / 2), Y0 + 40, 'khung lỗi: mất tay hoặc vai', { size: 13, fill: C.red, anchor: 'middle' });
      s.pops = [
        text(svg, s.X((s.ints[0][0] + s.ints[0][1]) / 2), Y1 - 22, 'Cảm ơn · 98%', { size: 22, w: 700, fill: C.green, anchor: 'middle' }),
        text(svg, s.X((a2 + b2) / 2), Y1 - 22, 'Yêu cầu ký lại', { size: 22, w: 700, fill: C.red, anchor: 'middle' }),
      ];
      return s;
    },
    step(s, t) {
      const u = t % 11;
      const cx = clamp(u / 9.5, 0, 1);
      let d = '';
      for (let x = 0; x <= cx; x += 0.004) d += `${d ? 'L' : 'M'}${s.X(x).toFixed(1)} ${s.Y(s.p(x)).toFixed(1)}`;
      set(s.curve, { d });
      set(s.cursor, { x1: s.X(cx), x2: s.X(cx), opacity: cx < 1 ? 1 : 0 });
      set(s.dot, { cx: s.X(cx), cy: s.Y(s.p(cx)), opacity: cx < 1 ? 1 : 0 });
      let state = 0, out = -1, queue = 'Hàng đợi: 1 khung (tư thế nghỉ)';
      s.ints.forEach(([a, b], i) => {
        set(s.shade[i], { x: s.X(a), width: Math.max(0, s.X(Math.min(cx, b)) - s.X(a)) * (cx > a ? 1 : 0) });
        if (cx > a && cx <= b) { state = 1; queue = `Hàng đợi: ${Math.round(((cx - a) / (b - a)) * 62)} khung`; }
        if (cx > b && cx < b + 0.08) { state = 2; out = i; queue = 'Đóng clip → chấm điểm'; }
        set(s.pops[i], { opacity: ease(phase(cx, b + 0.01, b + 0.05)), transform: `translate(0 ${(1 - ease(phase(cx, b + 0.01, b + 0.05))) * 10})` });
      });
      s.states.forEach((b, i) => set(b.r, { 'stroke-width': i === state ? 5 : 2, fill: tint(b.col, i === state ? 0.7 : 0.92) }));
      s.outs.forEach((b, i) => set(b.r, { 'stroke-width': i === out ? 5 : 2, fill: tint(b.col, i === out ? 0.7 : 0.94) }));
      s.queue.textContent = queue;
      s.ticks.forEach((k) => set(k.e, { opacity: cx > k.x ? 1 : 0 }));
      set(s.tickLabel, { opacity: cx > s.ints[1][0] + 0.03 ? 1 : 0 });
    },
  };

  // ================================================================ nguyên lý GCN
  KINDS.gcn = {
    build(svg) {
      vb(svg, 760, 420);
      const s = {};
      s.sk = skeleton(svg);
      s.d = hops(WRIST);
      s.nb = s.d.map((v, i) => (v === 1 ? i : -1)).filter((i) => i >= 0);
      s.ps = particles(svg, 8, C.blue, 5);
      divider(svg);
      s.R = el(svg, 'g');
      s.minis = [0.1, 0.45, 0.8].map((sv, k) => {
        const g = el(s.R, 'g', { transform: `translate(${418 + k * 112},110) scale(0.3)` });
        const m = skeleton(g, { bw: 8 });
        m.update(pose(sv));
        m.nodes[WRIST].setAttribute('fill', C.orange);
        m.nodes[WRIST].setAttribute('r', 14);
        text(s.R, 418 + k * 112 + 60, 260, ['t − 1', 't', 't + 1'][k], { size: 16, w: 600, fill: C.muted, anchor: 'middle' });
        return { k, P: pose(sv) };
      });
      s.title = title(svg, 16, 30);
      s.sub = text(svg, 16, 404, '', { size: 14, fill: C.muted });
      s.rt = text(svg, 580, 330, 'cổ tay ở khung t gom thêm', { size: 14, fill: C.muted, anchor: 'middle' });
      text(s.R, 580, 350, 'chính nó ở khung trước và sau', { size: 14, fill: C.muted, anchor: 'middle' });
      return s;
    },
    step(s, t) {
      const u = t % 11;
      const sk = s.sk, P = BASE;
      sk.reset();
      s.ps.forEach((p) => set(p, { opacity: 0 }));
      const col = [C.orange, C.blue, '#6ea6e6', '#a9c8f0', '#d3e3f7'];
      let k;
      if (u < 3.6) {
        s.title.textContent = '① Mỗi khớp gom thông tin từ khớp nối với nó';
        s.sub.textContent = 'cổ tay (cam) cộng thông tin của khuỷu tay và các ngón (xanh)';
        const p = (u % 1.8) / 1.8;
        s.nb.forEach((j, n) => {
          set(sk.nodes[j], { fill: C.blue });
          set(s.ps[n], { opacity: Math.sin(Math.PI * p), cx: lerp(P[j][0], P[WRIST][0], ease(p)), cy: lerp(P[j][1], P[WRIST][1], ease(p)) });
        });
        BONES.forEach(([a, b], e) => { if (a === WRIST || b === WRIST) set(sk.edges[e], { stroke: '#9cc2ee' }); });
        set(sk.nodes[WRIST], { fill: C.orange, r: 7 + 5 * ease(p) });
        set(s.R, { opacity: 0.25 });
      } else {
        k = u < 7.4 ? 1 + Math.min(2, Math.floor((u - 3.6) / 1.27)) : 3;
        s.title.textContent = u < 7.4 ? `② Xếp ${k} lớp: cổ tay “nhìn thấy” xa ${k} bước` : '③ Thêm chiều thời gian';
        s.sub.textContent = u < 7.4 ? 'càng nhiều lớp, mỗi khớp càng hiểu rộng ra cơ thể' : 'đồ thị không gian + tích chập theo thời gian';
        sk.nodes.forEach((n, i) => { if (s.d[i] <= k) set(n, { fill: col[s.d[i]], r: i === WRIST ? 11 : i >= 15 ? 5 : 8 }); });
        BONES.forEach(([a, b], e) => { if (s.d[a] <= k && s.d[b] <= k) set(sk.edges[e], { stroke: '#9cc2ee' }); });
        set(s.R, { opacity: u < 7.4 ? 0.25 : 1 });
        if (u >= 7.4) {
          const p = ((u - 7.4) % 1.8) / 1.8;
          const at = (m) => [418 + m.k * 112 + m.P[WRIST][0] * 0.3, 110 + m.P[WRIST][1] * 0.3];
          const c = at(s.minis[1]);
          [s.minis[0], s.minis[2]].forEach((m, n) => {
            const a = at(m);
            set(s.ps[6 + n], { fill: C.orange, opacity: Math.sin(Math.PI * p), cx: lerp(a[0], c[0], ease(p)), cy: lerp(a[1], c[1], ease(p)) });
          });
        }
      }
    },
  };

  // ================================================================ MS-TCN
  KINDS.mstcn = {
    build(svg) {
      vb(svg, 760, 420);
      const s = {};
      s.sk = skeleton(svg);
      s.sk.edges.forEach((e) => set(e, { 'stroke-dasharray': '2 10', opacity: 0.6 }));
      set(s.sk.head, { opacity: 0.3 });
      // cột vector sau khi gộp khớp
      s.col = [];
      for (let i = 0; i < 14; i++) s.col.push(el(svg, 'rect', { x: 360, y: 92 + i * 18, width: 22, height: 14, rx: 2, fill: tint(C.blue, 0.6) }));
      text(svg, 371, 80, '1 vector', { size: 13, w: 600, fill: C.blue, anchor: 'middle' });
      s.src = [0, 5, 6, 8, 10, 12, 22, 11, 9, 3, 24, 7, 13, 17];
      s.ps = particles(svg, 14, C.blue, 4);
      // lưới thời gian
      const R = 10, CN = 22, x0 = 420, y0 = 92, cw = 14, ch = 18;
      Object.assign(s, { R, CN, x0, y0, cw, ch });
      for (let r = 0; r < R; r++) for (let c = 0; c < CN; c++) {
        const env = Math.exp(-Math.pow((c - 11) / 5.5, 2));
        const v = env * (0.5 + 0.5 * Math.sin(c * 0.55 + r * 0.9));
        el(svg, 'rect', { x: x0 + c * cw, y: y0 + r * ch, width: cw - 2, height: ch - 2, fill: C.blue, opacity: 0.07 + 0.75 * v });
      }
      text(svg, x0, y0 - 14, 'mỗi cột là một khung →', { size: 13, fill: C.faint });
      s.taps = [0, 1, 2].map(() => el(svg, 'rect', { y: y0 - 4, height: R * ch + 6, width: cw + 2, fill: 'none', stroke: C.orange, 'stroke-width': 3, rx: 3 }));
      s.out = [];
      for (let c = 0; c < CN; c++) s.out.push(el(svg, 'rect', { x: x0 + c * cw, y: 330, width: cw - 2, height: 16, fill: C.orange, opacity: 0.08 }));
      s.links = [0, 1, 2].map(() => el(svg, 'line', { stroke: C.orange, 'stroke-width': 2 }));
      s.title = title(svg, 16, 30);
      s.title.textContent = 'Không có đồ thị: chỉ nhìn theo thời gian';
      s.dl = text(svg, 580, 380, '', { size: 16, w: 700, fill: C.orange, anchor: 'middle' });
      text(svg, 580, 402, 'các mức giãn 1, 2, 4, 8, 16 → nhìn được 125 khung', { size: 13, fill: C.muted, anchor: 'middle' });
      text(svg, 200, 404, '76 khớp gộp thành 1 vector mỗi khung', { size: 14, fill: C.muted, anchor: 'middle' });
      return s;
    },
    step(s, t) {
      s.sk.update(pose(wave(t, 5)));
      const P = s.sk.P;
      s.ps.forEach((p, k) => {
        const u = (t * 0.7 + k / 14) % 1;
        const a = P[s.src[k]], b = [371, 99 + k * 18];
        set(p, { cx: lerp(a[0], b[0], ease(u)), cy: lerp(a[1], b[1], ease(u)), opacity: Math.sin(Math.PI * u) });
      });
      const D = [1, 2, 4, 8], per = 2.6;
      const d = D[Math.floor(t / per) % 4];
      const p = (t % per) / per;
      const c = d + Math.round(p * (s.CN - 1 - 2 * d));
      [c - d, c, c + d].forEach((col, i) => {
        set(s.taps[i], { x: s.x0 + col * s.cw - 2 });
        set(s.links[i], { x1: s.x0 + col * s.cw + s.cw / 2 - 1, y1: s.y0 + s.R * s.ch + 2, x2: s.x0 + c * s.cw + s.cw / 2 - 1, y2: 330 });
      });
      s.out.forEach((o, i) => set(o, { opacity: i === c ? 1 : i < c ? 0.45 : 0.08 }));
      s.dl.textContent = `tích chập giãn d = ${d}`;
    },
  };

  // ================================================================ ST-GCN
  KINDS.stgcn = {
    build(svg) {
      vb(svg, 760, 420);
      const s = { sk: skeleton(svg), d: hops(WRIST) };
      s.title = title(svg, 16, 30);
      s.title.textContent = 'Đồ thị cố định: chỉ đi theo xương';
      text(svg, 200, 404, 'mỗi lớp lan thêm đúng 1 bước dọc theo xương', { size: 14, fill: C.muted, anchor: 'middle' });
      divider(svg);
      const N = 26, x0 = 424, x1 = 736;
      s.fr = [];
      for (let c = 0; c < N; c++) {
        const h = 0.5 - 0.5 * Math.cos((2 * Math.PI * c) / (N - 1));
        const x = lerp(x0, x1, c / (N - 1)), y = 270 - 150 * h;
        s.fr.push(el(svg, 'circle', { cx: x, cy: y, r: 5, fill: C.gray }));
      }
      text(svg, 424, 92, 'độ cao cổ tay qua các khung', { size: 13, fill: C.faint });
      el(svg, 'line', { x1: x0, y1: 290, x2: x1, y2: 290, stroke: C.line, 'stroke-width': 1.5 });
      s.win = el(svg, 'rect', { y: 100, height: 196, rx: 8, fill: tint(C.orange, 0.85), stroke: C.orange, 'stroke-width': 2, opacity: 0.8 });
      svg.insertBefore(s.win, s.fr[0]);
      s.out = [];
      for (let c = 0; c < N; c++) s.out.push(el(svg, 'rect', { x: lerp(x0, x1, c / (N - 1)) - 5, y: 318, width: 10, height: 14, rx: 2, fill: C.orange, opacity: 0.1 }));
      Object.assign(s, { N, x0, x1 });
      text(svg, 580, 370, 'tích chập thời gian · cửa sổ 9 khung', { size: 16, w: 700, fill: C.orange, anchor: 'middle' });
      return s;
    },
    step(s, t) {
      s.sk.update(pose(wave(t, 5)));
      s.sk.reset();
      const front = ((t % 2.6) / 2.6) * 7;
      s.sk.nodes.forEach((n, i) => {
        const a = clamp(1 - Math.abs(s.d[i] - front));
        if (s.d[i] <= front) set(n, { fill: mix(C.blue, C.orange, a), r: (i >= 15 ? 4 : 7) + 4 * a });
      });
      BONES.forEach(([a, b], e) => { if (s.d[a] <= front && s.d[b] <= front) set(s.sk.edges[e], { stroke: '#8fb9ea' }); });
      const k = Math.floor((t * 5) % (s.N - 8)) + 4;
      const xs = (c) => lerp(s.x0, s.x1, c / (s.N - 1));
      set(s.win, { x: xs(k - 4) - 9, width: xs(k + 4) - xs(k - 4) + 18 });
      s.fr.forEach((f, c) => set(f, { fill: Math.abs(c - k) <= 4 ? C.orange : C.gray }));
      s.out.forEach((o, c) => set(o, { opacity: c === k ? 1 : c < k ? 0.45 : 0.1 }));
    },
  };

  // ================================================================ CTR-GCN
  KINDS.ctrgcn = {
    build(svg) {
      vb(svg, 760, 420);
      const s = { sk: skeleton(svg) };
      s.ch = [
        { name: 'Kênh A · hình dạng bàn tay', color: C.blue, e: [[20, 21], [21, 22], [22, 23], [23, 24], [20, 24], [15, 19], [16, 18]] },
        { name: 'Kênh B · tay so với mặt', color: C.orange, e: [[12, 5], [22, 0], [12, 4], [21, 2]] },
        { name: 'Kênh C · hai tay với nhau', color: C.green, e: [[11, 12], [17, 22], [9, 10]] },
      ];
      s.ch.forEach((c) => { c.g = el(s.sk.extra, 'g', { opacity: 0 }); c.paths = c.e.map(() => el(c.g, 'path', { fill: 'none', stroke: c.color, 'stroke-width': 4, 'stroke-linecap': 'round' })); });
      divider(svg);
      // ma trận kề: dùng chung + tinh chỉnh từng kênh
      const n = 7, cell = 13;
      const mat = (x, y, label, color, extra) => {
        const g = el(svg, 'g');
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
          const base = i === j || Math.abs(i - j) === 1;
          const add = extra.some(([a, b]) => (a === i && b === j) || (a === j && b === i));
          el(g, 'rect', { x: x + j * cell, y: y + i * cell, width: cell - 1.5, height: cell - 1.5, fill: add ? color : base ? '#8a94a0' : '#eef0f3' });
        }
        const frame = el(g, 'rect', { x: x - 5, y: y - 5, width: n * cell + 8, height: n * cell + 8, rx: 5, fill: 'none', stroke: color, 'stroke-width': 1 });
        text(g, x + (n * cell) / 2, y + n * cell + 22, label, { size: 13, w: 600, fill: color, anchor: 'middle' });
        return frame;
      };
      mat(538, 56, 'Topo chung', C.faint, []);
      s.frames = [
        mat(430, 222, 'Kênh A', C.blue, [[4, 6], [5, 6], [3, 6]]),
        mat(538, 222, 'Kênh B', C.orange, [[0, 5], [1, 6], [0, 4]]),
        mat(646, 222, 'Kênh C', C.green, [[2, 5], [1, 4]]),
      ];
      [476, 584, 692].forEach((x) => arrow(svg, 584, 178, x, 214, C.faint, 1.5));
      text(svg, 600, 200, '+ tinh chỉnh', { size: 12, fill: C.faint });
      s.title = title(svg, 16, 30);
      s.sub = text(svg, 16, 404, '', { size: 15, w: 600 });
      return s;
    },
    step(s, t) {
      const P = pose(0.55 + 0.45 * Math.sin(t * 0.9) * 0.6);
      s.sk.update(P);
      const per = 2.6, cur = Math.floor(t / per) % 3, f = (t % per) / per;
      const a = Math.min(1, f * 5, (1 - f) * 5);
      s.ch.forEach((c, i) => {
        set(c.g, { opacity: i === cur ? a : 0.07 });
        c.e.forEach(([u, v], k) => {
          const A = P[u], B = P[v];
          const mx = (A[0] + B[0]) / 2, my = (A[1] + B[1]) / 2;
          const dx = B[0] - A[0], dy = B[1] - A[1], L = Math.hypot(dx, dy) || 1;
          const bend = Math.min(40, L * 0.25);
          set(c.paths[k], { d: `M${A[0]} ${A[1]}Q${mx - (dy / L) * bend} ${my + (dx / L) * bend} ${B[0]} ${B[1]}` });
        });
      });
      s.frames.forEach((fr, i) => set(fr, { 'stroke-width': i === cur ? 3.5 : 1 }));
      s.title.textContent = 'Mỗi kênh học thêm cạnh riêng';
      s.sub.textContent = s.ch[cur].name;
      set(s.sub, { fill: s.ch[cur].color });
    },
  };

  // ================================================================ AAGCN
  KINDS.aagcn = {
    build(svg) {
      vb(svg, 760, 420);
      const s = { sk: skeleton(svg) };
      s.face = [0, 1, 2, 3, 4, 5, 6];
      s.inf = s.face.map(() => el(s.sk.extra, 'line', { stroke: C.purple, 'stroke-linecap': 'round' }));
      divider(svg);
      const grp = (y, label) => text(svg, 420, y, label, { size: 14, w: 700, fill: C.purple });
      grp(70, 'Chú ý theo khớp');
      s.jl = ['Cổ tay phải', 'Ngón tay phải', 'Khuỷu phải', 'Mặt', 'Tay trái'];
      s.jb = s.jl.map((l, i) => { text(svg, 420, 96 + i * 20, l, { size: 12, fill: C.muted }); return el(svg, 'rect', { x: 520, y: 86 + i * 20, height: 12, rx: 3, fill: C.purple }); });
      grp(222, 'Chú ý theo khung');
      s.fb = [];
      for (let c = 0; c < 22; c++) s.fb.push(el(svg, 'rect', { x: 420 + c * 14, width: 10, rx: 2, fill: tint(C.purple, 0.5) }));
      s.cur = el(svg, 'rect', { y: 232, width: 14, height: 70, fill: 'none', stroke: C.purple, 'stroke-width': 2, rx: 3 });
      grp(330, 'Chú ý theo kênh');
      s.cb = [];
      for (let c = 0; c < 16; c++) s.cb.push(el(svg, 'rect', { x: 420 + c * 19, width: 14, rx: 2, fill: tint(C.purple, 0.3) }));
      s.title = title(svg, 16, 30);
      s.title.textContent = 'Đồ thị tự suy ra từ chính clip';
      text(svg, 200, 404, 'cạnh tím đổi theo tư thế · khớp lớn = được chú ý', { size: 14, fill: C.muted, anchor: 'middle' });
      return s;
    },
    step(s, t) {
      const P = pose(wave(t, 5)), Q = pose(wave(t - 0.06, 5));
      s.sk.update(P);
      s.sk.reset();
      const sp = P.map((p, i) => dist(p, Q[i]));
      s.sk.nodes.forEach((n, i) => { const a = clamp(sp[i] / 6); set(n, { r: (i >= 15 ? 4 : 7) + 9 * a, fill: mix(C.gray, C.purple, a * 1.4) }); });
      const w = softmax(s.face.map((j) => -dist(P[WRIST], P[j]) / 28));
      s.face.forEach((j, k) => set(s.inf[k], { x1: P[WRIST][0], y1: P[WRIST][1], x2: P[j][0], y2: P[j][1], 'stroke-width': 1 + 7 * w[k], opacity: 0.15 + 0.85 * w[k] }));
      const prox = clamp(1 - dist(P[WRIST], P[5]) / 180);
      const jv = [sp[12] / 6, sp[22] / 6, sp[10] / 6, prox, 0.06];
      s.jb.forEach((b, i) => set(b, { width: 8 + 200 * clamp(jv[i]) }));
      const u = (t % 5) / 5;
      s.fb.forEach((b, c) => { const v = Math.abs(Math.sin((2 * Math.PI * c) / 22)); set(b, { y: 300 - 66 * v, height: 66 * v + 2 }); });
      set(s.cur, { x: 418 + Math.floor(u * 22) * 14 });
      s.cb.forEach((b, c) => { const v = 0.25 + 0.7 * Math.abs(Math.sin(t * 0.8 + c * 0.9)); set(b, { y: 400 - 56 * v, height: 56 * v }); });
    },
  };

  // ================================================================ Transformer
  KINDS.transformer = {
    build(svg) {
      vb(svg, 760, 420);
      const s = { sk: skeleton(svg) };
      s.sk.g.setAttribute('opacity', 0.55);
      s.box = el(svg, 'rect', { x: 352, y: 196, width: 34, height: 34, rx: 6, fill: C.orange });
      text(svg, 369, 252, '1 token', { size: 12, w: 600, fill: C.orange, anchor: 'middle' });
      s.ps = particles(svg, 10, C.orange, 4);
      s.src = [0, 6, 8, 10, 12, 22, 11, 7, 3, 24];
      divider(svg);
      const N = 12, x0 = 428, gap = 27, ty = 330;
      Object.assign(s, { N, x0, gap, ty });
      s.arcs = []; s.cls = [];
      s.clsBox = el(svg, 'rect', { x: 562, y: 70, width: 44, height: 30, rx: 6, fill: C.navy });
      text(svg, 584, 90, 'CLS', { size: 12, w: 700, fill: '#fff', anchor: 'middle' });
      for (let j = 0; j < N; j++) s.cls.push(el(svg, 'line', { x1: 584, y1: 100, x2: x0 + j * gap + 10, y2: ty, stroke: C.navy, 'stroke-width': 1.5 }));
      for (let j = 0; j < N; j++) s.arcs.push(el(svg, 'path', { fill: 'none', stroke: C.orange, 'stroke-linecap': 'round' }));
      s.tok = [];
      for (let j = 0; j < N; j++) s.tok.push(el(svg, 'rect', { x: x0 + j * gap, y: ty, width: 20, height: 20, rx: 4, fill: tint(C.blue, 0.4) }));
      text(svg, 584, ty + 46, 'mỗi ô là một khung · thời gian →', { size: 13, fill: C.faint, anchor: 'middle' });
      s.title = title(svg, 16, 30);
      s.title.textContent = 'Mọi khung nhìn mọi khung';
      text(svg, 200, 404, 'không dùng cấu trúc cơ thể: phải tự học mọi quan hệ', { size: 14, fill: C.muted, anchor: 'middle' });
      return s;
    },
    step(s, t) {
      s.sk.update(pose(wave(t, 5)));
      const P = s.sk.P;
      s.ps.forEach((p, k) => {
        const u = (t * 0.8 + k / 10) % 1;
        const a = P[s.src[k]];
        set(p, { cx: lerp(a[0], 369, ease(u)), cy: lerp(a[1], 213, ease(u)), opacity: Math.sin(Math.PI * u) });
      });
      const motion = (j) => Math.pow(Math.sin((Math.PI * j) / (s.N - 1)), 2);
      const q = Math.floor(t / 1.3) % s.N;
      const raw = [];
      for (let j = 0; j < s.N; j++) raw.push(-Math.abs(j - q) / 2.5 + 1.6 * motion(j));
      const w = softmax(raw);
      const cx = (j) => s.x0 + j * s.gap + 10;
      s.arcs.forEach((a, j) => {
        if (j === q) { set(a, { opacity: 0 }); return; }
        const h = 26 + Math.abs(j - q) * 13;
        set(a, { d: `M${cx(q)} ${s.ty}Q${(cx(q) + cx(j)) / 2} ${s.ty - h} ${cx(j)} ${s.ty}`, 'stroke-width': 1 + 9 * w[j], opacity: clamp(0.15 + 3 * w[j]) });
      });
      s.tok.forEach((b, j) => set(b, { fill: j === q ? C.orange : tint(C.blue, 0.75 - 0.6 * motion(j)) }));
      s.cls.forEach((l, j) => set(l, { opacity: 0.08 + 0.6 * motion(j) }));
    },
  };

  // ================================================================ MultiRate
  KINDS.multirate = {
    build(svg) {
      vb(svg, 760, 420);
      const s = { sk: skeleton(svg) };
      divider(svg);
      const rows = [[32, 8, C.blue, 1], [64, 16, C.orange, 0.5], [96, 24, C.green, 0.25]];
      const xa = 470, xb = 640;
      const h = (u) => Math.exp(-Math.pow((u - 0.5) / 0.22, 4));
      s.rows = rows.map(([len, n, col, width], r) => {
        const y = 86 + r * 92;
        text(svg, 420, y + 30, `${len}`, { size: 18, w: 700, fill: col });
        text(svg, 420, y + 48, 'khung', { size: 12, fill: C.faint });
        let d = '';
        for (let k = 0; k <= 60; k++) { const u = k / 60; d += `${k ? 'L' : 'M'}${lerp(xa, xb, u)} ${y + 50 - 44 * h(u)}`; }
        el(svg, 'path', { d, fill: 'none', stroke: C.line, 'stroke-width': 2 });
        const dots = [];
        for (let k = 0; k < n; k++) { const u = (k + 0.5) / n; dots.push({ u, e: el(svg, 'circle', { cx: lerp(xa, xb, u), cy: y + 50 - 44 * h(u), r: 4.5, fill: C.gray }) }); }
        const bh = 14 + 44 * width;
        el(svg, 'rect', { x: 662, y: y + 30 - bh / 2, width: 54, height: bh, rx: 6, fill: tint(col, 0.75), stroke: col, 'stroke-width': 1.5 });
        text(svg, 689, y + 34, 'ST-GCN', { size: 11, w: 700, fill: C.ink, anchor: 'middle' });
        arrow(svg, 716, y + 30, 744, y + 30, col, 1.5);
        return { dots, col };
      });
      s.play = el(svg, 'line', { y1: 70, y2: 330, stroke: C.navy, 'stroke-width': 1.5, 'stroke-dasharray': '3 4' });
      Object.assign(s, { xa, xb });
      text(svg, 420, 372, 'Attention gộp', { size: 14, w: 700, fill: C.navy });
      s.fw = rows.map(([len, , col], r) => el(svg, 'rect', { x: 530, y: 356 + r * 14, height: 10, rx: 3, fill: col }));
      s.title = title(svg, 16, 30);
      s.title.textContent = 'Xem cùng một ký hiệu ở ba tốc độ';
      text(svg, 200, 404, 'nhánh 96 khung hẹp hơn (¼) để không tốn gấp ba', { size: 14, fill: C.muted, anchor: 'middle' });
      return s;
    },
    step(s, t) {
      s.sk.update(pose(wave(t, 3)));
      const u = (t % 3) / 3;
      set(s.play, { x1: lerp(s.xa, s.xb, u), x2: lerp(s.xa, s.xb, u) });
      s.rows.forEach((r) => r.dots.forEach((d) => set(d.e, { fill: d.u <= u ? r.col : C.gray, r: Math.abs(d.u - u) < 0.04 ? 7 : 4.5 })));
      const w = softmax([Math.sin(t * 0.9), Math.sin(t * 0.9 + 2.1), Math.sin(t * 0.9 + 4.2)].map((x) => 1.6 * x));
      s.fw.forEach((b, i) => set(b, { width: 8 + 190 * w[i] }));
    },
  };

  // ================================================================ AnchorGCN
  KINDS.anchor = {
    build(svg) {
      vb(svg, 760, 420);
      const s = { sk: skeleton(svg) };
      s.anchors = [
        ['tai trái', (P) => P[3]], ['tai phải', (P) => P[4]], ['tâm mắt', (P) => [(P[1][0] + P[2][0]) / 2, (P[1][1] + P[2][1]) / 2]],
        ['miệng', (P) => P[5]], ['tâm đầu', (P) => [(P[3][0] + P[4][0]) / 2, P[3][1] - 22]], ['cổ', (P) => P[6]],
        ['vai trái', (P) => P[7]], ['vai phải', (P) => P[8]],
      ];
      s.lines = s.anchors.map(() => el(s.sk.extra, 'line', { stroke: C.orange, 'stroke-dasharray': '6 5', 'stroke-linecap': 'round' }));
      s.marks = s.anchors.map(() => el(svg, 'path', { fill: C.navy }));
      divider(svg);
      text(svg, 420, 70, 'Bàn tay đang gần mốc nào?', { size: 15, w: 700, fill: C.navy });
      s.bars = s.anchors.map(([n], i) => { text(svg, 420, 104 + i * 34, n, { size: 13, fill: C.muted }); el(svg, 'rect', { x: 500, y: 92 + i * 34, width: 230, height: 14, rx: 4, fill: '#eef0f3' }); return el(svg, 'rect', { x: 500, y: 92 + i * 34, height: 14, rx: 4, fill: C.orange }); });
      s.title = title(svg, 16, 30);
      s.title.textContent = 'Đo tay so với 8 mốc trên mặt và thân';
      text(svg, 200, 404, 'không cần đi ~10 bước dọc xương từ ngón tay tới miệng', { size: 14, fill: C.muted, anchor: 'middle' });
      return s;
    },
    step(s, t) {
      const P = pose(0.15 + 0.85 * wave(t, 6));
      s.sk.update(P);
      const hand = P[22];
      const pts = s.anchors.map(([, f]) => f(P));
      const w = softmax(pts.map((p) => -dist(hand, p) / 26));
      const top = w.indexOf(Math.max(...w));
      pts.forEach((p, i) => {
        set(s.lines[i], { x1: hand[0], y1: hand[1], x2: p[0], y2: p[1], 'stroke-width': 1 + 6 * w[i], opacity: 0.18 + 0.82 * w[i] });
        const r = i === top ? 9 : 6;
        set(s.marks[i], { d: `M${p[0]} ${p[1] - r}L${p[0] + r} ${p[1]}L${p[0]} ${p[1] + r}L${p[0] - r} ${p[1]}z`, fill: i === top ? C.orange : C.navy });
        set(s.bars[i], { width: 4 + 226 * w[i] });
      });
    },
  };

  // ---------------------------------------------------------------- vòng lặp
  let active = [], t0 = performance.now();
  // ?at=6 cố định thời điểm hình động (để chụp kiểm tra); bình thường không dùng.
  const AT = parseFloat(new URLSearchParams(location.search).get('at'));
  function prepare(svg) {
    if (!svg._st) {
      const kind = KINDS[svg.dataset.kind];
      if (!kind) return null;
      svg._st = kind.build(svg);
    }
    return svg;
  }
  function activate(slide) {
    active = [...slide.querySelectorAll('svg.anim')].map(prepare).filter(Boolean);
    t0 = performance.now();
  }
  function loop(now) {
    const t = Number.isFinite(AT) ? AT : (now - t0) / 1000;
    for (const svg of active) KINDS[svg.dataset.kind].step(svg._st, t);
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);
  window.SlideAnim = { activate };
})();
