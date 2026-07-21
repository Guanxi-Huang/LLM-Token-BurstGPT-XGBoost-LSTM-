import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile, layers, shape, text } from "@oai/artifact-tool";

const SLIDE = { width: 1280, height: 720 };
const C = {
  navy: "#0B1F3A",
  blue: "#1D4ED8",
  cyan: "#0EA5E9",
  orange: "#F97316",
  green: "#15803D",
  red: "#B91C1C",
  ink: "#172033",
  muted: "#5B6573",
  pale: "#F3F6FA",
  paleBlue: "#EAF2FF",
  border: "#D8E0EA",
  white: "#FFFFFF",
};
const FONT = "Microsoft YaHei";

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i += 2) {
    const key = argv[i]?.replace(/^--/, "");
    const value = argv[i + 1];
    if (key && value) out[key] = value;
  }
  return out;
}

function tx(value, x, y, w, h, size, color = C.ink, bold = false, options = {}) {
  return text([value], {
    name: options.name,
    position: { left: x, top: y },
    width: w,
    height: h,
    style: {
      fontSize: `${size}px`,
      typeface: FONT,
      color,
      bold,
      alignment: options.align || "left",
      verticalAlignment: options.valign || "middle",
      autoFit: "shrinkText",
      insets: { top: 0, right: 0, bottom: 0, left: 0 },
    },
  });
}

function box(x, y, w, h, fill = C.white, stroke = C.border, radius = "rounded-xl") {
  return shape({
    geometry: "roundRect",
    fill,
    line: { style: "solid", fill: stroke, width: 1 },
    borderRadius: radius,
    position: { left: x, top: y },
    width: w,
    height: h,
  });
}

function line(x, y, w, color = C.border, width = 2) {
  return shape({
    geometry: "straightConnector1",
    fill: "none",
    line: { style: "solid", fill: color, width },
    position: { left: x, top: y },
    width: w,
    height: 0.01,
  });
}

function dot(x, y, color = C.blue, r = 9) {
  return shape({ geometry: "ellipse", fill: color, line: { fill: color, width: 0 }, position: { left: x, top: y }, width: r, height: r });
}

function compose(slide, elements) {
  slide.background.fill = C.white;
  slide.compose(layers({ width: "fill", height: "fill" }, elements), {
    frame: { left: 0, top: 0, width: SLIDE.width, height: SLIDE.height },
    baseUnit: 1,
  });
}

function header(title, number, eyebrow = "BURSTGPT · 15 MIN") {
  return [
    shape({ geometry: "rect", fill: C.blue, line: { fill: C.blue, width: 0 }, position: { left: 0, top: 0 }, width: 1280, height: 9 }),
    tx(eyebrow, 48, 28, 420, 24, 13, C.blue, true),
    tx(title, 48, 57, 1135, 66, 38, C.navy, true),
    tx(String(number).padStart(2, "0"), 1184, 660, 48, 24, 13, C.muted, true, { align: "right" }),
  ];
}

function footer(source) {
  return [line(48, 650, 1138, C.border, 1), tx(source, 48, 661, 980, 18, 11, C.muted, false)];
}

async function imageBytes(file) {
  const bytes = await fs.readFile(file);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function addImage(slide, bytes, position, alt) {
  slide.images.add({
    blob: bytes,
    contentType: "image/png",
    alt,
    fit: "contain",
    position,
  });
}

function addNotes(slide, note) {
  slide.speakerNotes.textFrame.setText(note);
  slide.speakerNotes.setVisible(true);
}

function addCover(p, note) {
  const s = p.slides.add();
  compose(s, [
    shape({ geometry: "rect", fill: C.navy, line: { fill: C.navy, width: 0 }, position: { left: 0, top: 0 }, width: 1280, height: 720 }),
    shape({ geometry: "rect", fill: C.cyan, line: { fill: C.cyan, width: 0 }, position: { left: 0, top: 0 }, width: 18, height: 720 }),
    tx("BURSTGPT · 研究复现包", 62, 54, 500, 30, 16, "#8FD8FF", true),
    tx("LLM 服务的 Token 负载预测\n与突发流量识别", 62, 148, 1000, 205, 62, C.white, true),
    tx("XGBoost 与 LSTM 的无泄漏比较", 64, 380, 760, 48, 29, "#C9D7EA", false),
    line(64, 474, 785, "#46617E", 2),
    tx("结论先行", 64, 508, 130, 30, 16, "#8FD8FF", true),
    tx("平均误差可以改善；固定阈值极端突发仍难识别。", 64, 548, 980, 54, 28, C.white, true),
    tx("15 分钟展示 · 2026-07-21", 64, 657, 430, 24, 14, "#AFC1D7"),
  ]);
  addNotes(s, note);
}

function addQuestion(p, note) {
  const s = p.slides.add();
  const e = [...header("研究不是只问“谁的 MAE 最低”", 2)];
  e.push(box(48, 150, 390, 450, C.navy, C.navy));
  e.push(tx("研究问题", 76, 178, 150, 28, 16, "#8FD8FF", true));
  e.push(tx("在严格避免未来信息后，\n历史请求轨迹能否：", 76, 232, 320, 80, 27, C.white, true));
  ["预测未来 15 分钟 Token 总量", "识别训练期 P95 容量超限", "跨时期保持模型排名"].forEach((v, i) => {
    e.push(dot(78, 360 + i * 70, i === 1 ? C.orange : C.cyan, 10));
    e.push(tx(v, 101, 343 + i * 70, 290, 48, 18, C.white, i === 1));
  });
  const cards = [
    ["01", "统一任务", "同一轨迹、同一时间契约下比较回归与阈值识别。"],
    ["02", "可审计防泄漏", "输入截止、目标时点、切分、scaler 与 P95 均有机器校验。"],
    ["03", "区分三类结论", "点预测、固定容量超限、跨时期迁移分别报告。"],
  ];
  cards.forEach((c, i) => {
    const y = 150 + i * 144;
    e.push(box(470, y, 718, 120, i === 1 ? C.paleBlue : C.pale, C.border));
    e.push(tx(c[0], 494, y + 25, 58, 50, 28, i === 1 ? C.blue : C.muted, true));
    e.push(tx(c[1], 572, y + 18, 220, 34, 21, C.navy, true));
    e.push(tx(c[2], 572, y + 55, 565, 46, 16, C.muted));
  });
  e.push(...footer("来源：paper/paper_draft.md，第 1–2 节"));
  compose(s, e);
  addNotes(s, note);
}

function addData(p, note) {
  const s = p.slides.add();
  const e = [...header("主实验覆盖 5,058,910 条清洗后请求", 3)];
  const stats = [
    ["5,058,910", "主批次清洗后请求", C.blue],
    ["34,848", "连续 5 分钟窗口", C.cyan],
    ["315,246.2", "训练集 P95 · Token/5min", C.orange],
  ];
  stats.forEach((c, i) => {
    const x = 48 + i * 388;
    e.push(box(x, 158, 352, 205, i === 2 ? "#FFF5ED" : C.pale, C.border));
    e.push(shape({ geometry: "rect", fill: c[2], line: { fill: c[2], width: 0 }, position: { left: x, top: 158 }, width: 352, height: 9 }));
    e.push(tx(c[0], x + 24, 210, 304, 66, 36, C.navy, true, { align: "center" }));
    e.push(tx(c[1], x + 24, 292, 304, 36, 16, C.muted, false, { align: "center" }));
  });
  e.push(box(48, 401, 1128, 185, C.white, C.border));
  e.push(tx("请求级定义", 74, 425, 190, 30, 17, C.blue, true));
  e.push(tx("total_tokensᵢ = request_tokensᵢ + response_tokensᵢ", 74, 466, 470, 40, 21, C.navy, true));
  e.push(tx("窗口 [bₖ, bₖ+5min) 内求和；补齐零请求窗口", 74, 518, 500, 30, 16, C.muted));
  e.push(line(610, 424, 1, C.border, 1));
  e.push(tx("按目标时间连续切分", 648, 425, 230, 30, 17, C.blue, true));
  e.push(tx("70% 训练  ·  15% 验证  ·  15% 测试", 648, 469, 455, 35, 21, C.navy, true));
  e.push(tx("24,393  ·  5,227  ·  5,228 个目标窗口", 648, 518, 450, 30, 16, C.muted));
  e.push(...footer("来源：表 1、表 2；artifacts/final/tables"));
  compose(s, e);
  addNotes(s, note);
}

function addLeakage(p, note) {
  const s = p.slides.add();
  const e = [...header("预测原点 t 看不到 t 窗口", 4, "时间契约 · NO LEAKAGE")];
  e.push(tx("完整输入截止", 48, 164, 220, 28, 16, C.muted, true));
  e.push(tx("预测原点", 510, 164, 180, 28, 16, C.muted, true, { align: "center" }));
  e.push(tx("15 分钟目标", 964, 164, 220, 28, 16, C.muted, true, { align: "right" }));
  e.push(line(128, 300, 1024, C.navy, 4));
  const nodes = [
    [176, "t−5min", "最后可用的完整窗口", C.blue],
    [600, "t", "forecast origin", C.navy],
    [1060, "t+15min", "预测目标 / split key", C.orange],
  ];
  nodes.forEach((n) => {
    e.push(shape({ geometry: "ellipse", fill: C.white, line: { style: "solid", fill: n[3], width: 5 }, position: { left: n[0] - 22, top: 278 }, width: 44, height: 44 }));
    e.push(tx(n[1], n[0] - 105, 335, 210, 38, 24, n[3], true, { align: "center" }));
    e.push(tx(n[2], n[0] - 135, 378, 270, 35, 15, C.muted, false, { align: "center" }));
  });
  e.push(box(48, 462, 1128, 128, C.paleBlue, "#BFD4FA"));
  const checks = ["特征只来自 ≤t−5", "阈值只来自训练集", "scaler 只拟合训练输入", "调参只看验证集"];
  checks.forEach((v, i) => {
    const x = 78 + i * 274;
    e.push(shape({ geometry: "ellipse", fill: C.green, line: { fill: C.green, width: 0 }, position: { left: x, top: 500 }, width: 22, height: 22 }));
    e.push(tx("✓", x + 2, 499, 18, 23, 15, C.white, true, { align: "center" }));
    e.push(tx(v, x + 34, 488, 210, 46, 15, C.navy, true));
  });
  e.push(...footer("机器证据：scripts/validate_temporal_contract.py（PASS）"));
  compose(s, e);
  addNotes(s, note);
}

function addModels(p, note) {
  const s = p.slides.add();
  const e = [...header("四个模型共享同一目标与测试边界", 5)];
  const cards = [
    ["Persistence", "ŷ(t+h)=y(t−5)", "最强短期参照", C.navy],
    ["Seasonal Naive", "历史相对周期", "周期参照", C.cyan],
    ["XGBoost", "36 个历史特征", "12 个候选 · 验证 MAE", C.blue],
    ["LSTM", "144×10 序列", "32 units · 5,537 参数", C.orange],
  ];
  cards.forEach((c, i) => {
    const x = 48 + (i % 2) * 574;
    const y = 154 + Math.floor(i / 2) * 210;
    e.push(box(x, y, 542, 172, i >= 2 ? C.paleBlue : C.pale, C.border));
    e.push(shape({ geometry: "rect", fill: c[3], line: { fill: c[3], width: 0 }, position: { left: x, top: y }, width: 10, height: 172 }));
    e.push(tx(c[0], x + 34, y + 25, 280, 34, 23, C.navy, true));
    e.push(tx(c[1], x + 34, y + 72, 450, 34, 19, c[3], true));
    e.push(tx(c[2], x + 34, y + 117, 450, 27, 15, C.muted));
  });
  e.push(box(48, 588, 1128, 47, "#F8FAFC", C.border));
  e.push(tx("共同原则：验证集负责选择；冻结后测试只预测一次。", 78, 595, 1070, 31, 17, C.navy, true, { align: "center" }));
  e.push(...footer("来源：表 3、表 4；paper/paper_draft.md，第 3.5–3.6 节"));
  compose(s, e);
  addNotes(s, note);
}

function addMainResult(p, note, fig) {
  const s = p.slides.add();
  const e = [...header("主测试：LSTM 的 15 分钟 MAE 最低", 6, "结果 1 · 点预测")];
  e.push(box(42, 143, 846, 447, C.white, C.border));
  e.push(box(917, 143, 273, 447, C.navy, C.navy));
  e.push(tx("11,054.5", 942, 188, 222, 58, 35, C.white, true, { align: "center" }));
  e.push(tx("Token/5min", 942, 250, 222, 28, 15, "#BFD0E5", false, { align: "center" }));
  e.push(tx("LSTM 测试 MAE", 942, 286, 222, 28, 17, "#8FD8FF", true, { align: "center" }));
  e.push(line(947, 341, 212, "#46617E", 1));
  e.push(tx("−10.4%", 942, 366, 222, 45, 30, "#8FD8FF", true, { align: "center" }));
  e.push(tx("相对 Persistence", 942, 413, 222, 27, 14, "#BFD0E5", false, { align: "center" }));
  e.push(tx("但优势只属于该时间段与点预测指标。", 947, 474, 212, 72, 16, C.white, true, { align: "center" }));
  e.push(...footer("图 13；artifacts/final/figures/fig_13_model_mae_f1_h15.png"));
  compose(s, e);
  addImage(s, fig, { left: 65, top: 171, width: 800, height: 388 }, "15 分钟主测试模型 MAE 与 F1 对比图");
  addNotes(s, note);
}

function addBurst(p, note, fig) {
  const s = p.slides.add();
  const e = [...header("固定 P95 下，四个模型都漏掉 5 个测试突发", 7, "结果 2 · 突发识别")];
  e.push(box(42, 143, 860, 445, C.white, C.border));
  e.push(box(928, 143, 262, 445, "#FFF5ED", "#FFD2B4"));
  e.push(tx("F1 = 0", 950, 178, 218, 48, 33, C.red, true, { align: "center" }));
  e.push(tx("四个模型一致", 950, 226, 218, 28, 15, C.muted, false, { align: "center" }));
  e.push(line(954, 279, 210, "#F0B58B", 1));
  e.push(tx("AP 最高", 950, 310, 218, 28, 16, C.muted, true, { align: "center" }));
  e.push(tx("XGBoost  0.0113", 950, 346, 218, 35, 23, C.orange, true, { align: "center" }));
  e.push(tx("LSTM  0.0108", 950, 393, 218, 29, 17, C.navy, true, { align: "center" }));
  e.push(tx("排序略有差异，但不足以支持可部署的固定阈值检测。", 950, 456, 218, 82, 15, C.ink, true, { align: "center" }));
  e.push(...footer("图 12；阈值 = 训练集 P95 = 315,246.2 Token/5min"));
  compose(s, e);
  addImage(s, fig, { left: 62, top: 171, width: 820, height: 390 }, "测试期突发窗口实际值与模型预测局部放大图");
  addNotes(s, note);
}

function addRobustness(p, note, fig) {
  const s = p.slides.add();
  const e = [...header("批次 3 零样本迁移中，Persistence 反而最好", 8, "结果 3 · 跨时期稳健性")];
  e.push(box(42, 143, 855, 430, C.white, C.border));
  e.push(box(925, 143, 265, 430, C.paleBlue, "#BFD4FA"));
  e.push(tx("43,838.2", 948, 177, 220, 48, 31, C.blue, true, { align: "center" }));
  e.push(tx("Persistence MAE", 948, 225, 220, 27, 15, C.muted, false, { align: "center" }));
  e.push(tx("F1  0.711", 948, 289, 220, 42, 28, C.green, true, { align: "center" }));
  e.push(line(954, 360, 208, "#BFD4FA", 1));
  e.push(tx("XGBoost", 948, 389, 105, 24, 15, C.muted, true));
  e.push(tx("61,795.3 · F1 0.509", 948, 418, 220, 28, 16, C.navy, true));
  e.push(tx("LSTM", 948, 466, 105, 24, 15, C.muted, true));
  e.push(tx("53,509.2 · F1 0.520", 948, 495, 220, 28, 16, C.navy, true));
  e.push(...footer("Week 5 图 10；批次 3 不重新拟合、不重新选参"));
  compose(s, e);
  addImage(s, fig, { left: 62, top: 170, width: 815, height: 372 }, "BurstGPT 批次 3 零样本跨时期稳健性对比图");
  addNotes(s, note);
}

function addLimits(p, note) {
  const s = p.slides.add();
  const e = [...header("结论成立，但边界必须和数字一起展示", 9)];
  const items = [
    ["极少正例", "主测试只有 5 个固定阈值突发窗，F1 高方差。", C.orange],
    ["时间语义缺失", "相对日/周相位不能解释为真实工作日或节假日。", C.cyan],
    ["清洗不确定性", "无 request_id；精确重复删除可能误删并发同值请求。", C.blue],
    ["模型处理不完全对称", "LSTM 负预测裁零，XGBoost 未裁零；已在论文披露。", C.navy],
  ];
  items.forEach((c, i) => {
    const x = 48 + (i % 2) * 574;
    const y = 155 + Math.floor(i / 2) * 206;
    e.push(box(x, y, 542, 166, C.pale, C.border));
    e.push(shape({ geometry: "ellipse", fill: c[2], line: { fill: c[2], width: 0 }, position: { left: x + 27, top: y + 29 }, width: 36, height: 36 }));
    e.push(tx(String(i + 1), x + 27, y + 29, 36, 36, 16, C.white, true, { align: "center" }));
    e.push(tx(c[0], x + 82, y + 25, 420, 34, 21, C.navy, true));
    e.push(tx(c[1], x + 82, y + 72, 420, 66, 16, C.muted));
  });
  e.push(box(48, 580, 1128, 54, "#FFF5ED", "#FFD2B4"));
  e.push(tx("不作过度推断：本研究未证明节能、SLO 改善或生产部署收益。", 72, 591, 1080, 32, 17, C.red, true, { align: "center" }));
  e.push(...footer("来源：paper/paper_draft.md，第 5 节"));
  compose(s, e);
  addNotes(s, note);
}

function addConclusion(p, note) {
  const s = p.slides.add();
  compose(s, [
    shape({ geometry: "rect", fill: C.navy, line: { fill: C.navy, width: 0 }, position: { left: 0, top: 0 }, width: 1280, height: 720 }),
    shape({ geometry: "rect", fill: C.cyan, line: { fill: C.cyan, width: 0 }, position: { left: 0, top: 0 }, width: 18, height: 720 }),
    tx("结论", 64, 52, 180, 30, 16, "#8FD8FF", true),
    tx("三个任务，三个答案", 64, 118, 980, 88, 52, C.white, true),
    tx("01", 72, 278, 60, 42, 25, "#8FD8FF", true),
    tx("点预测", 150, 274, 160, 40, 23, C.white, true),
    tx("LSTM 在主测试 MAE 最低。", 330, 274, 730, 40, 23, "#C9D7EA"),
    tx("02", 72, 365, 60, 42, 25, "#FFB781", true),
    tx("固定阈值突发", 150, 361, 200, 40, 23, C.white, true),
    tx("四模型 F1 均为 0；XGBoost AP 略高。", 370, 361, 750, 40, 23, "#C9D7EA"),
    tx("03", 72, 452, 60, 42, 25, "#8FD8FF", true),
    tx("跨时期迁移", 150, 448, 190, 40, 23, C.white, true),
    tx("Persistence 在批次 3 重新领先。", 360, 448, 720, 40, 23, "#C9D7EA"),
    line(64, 548, 1020, "#46617E", 2),
    tx("下一步：概率预测 · 自适应阈值 · 在线更新 · 多批次滚动评估", 64, 578, 1050, 48, 24, C.white, true),
    tx("代码、表图、实验日志与论文均可按 README 复跑", 64, 662, 760, 23, 14, "#AFC1D7"),
    tx("10 / 10", 1120, 662, 100, 23, 14, "#AFC1D7", true, { align: "right" }),
  ]);
  addNotes(s, note);
}

const notes = [
  "【约 1:00】大家好，这项研究关注 LLM 服务的短期 Token 负载，而不是只看请求数。Token 同时包含输入和输出长度，更接近推理系统面对的工作量。我先给结论：在主测试期，LSTM 的总体 MAE 最低；但当任务变成识别训练期 P95 以上的极端负载时，四个模型都没有命中测试突发；到了另一个时期，简单 Persistence 又重新领先。今天的重点不是宣布一个永久赢家，而是展示三个任务为什么必须分开评价，以及这些结论如何从代码、表格和日志追溯。",
  "【约 1:20】研究问题分三层。第一，历史 5 分钟窗口能否预测未来 15 分钟的 Token 总量；第二，同样的预测是否足以识别超过固定容量阈值的突发；第三，主时期选出的模型能否零样本迁移到批次 3。贡献也对应三点：把四个方法放在完全相同的时间契约下比较；用机器校验输入截止、目标时点、训练集拟合和验证集选参；最后不把 MAE、阈值分类和迁移稳定性混成一个指标。后面每个结果页只回答其中一个问题。",
  "【约 1:25】主实验使用 without_fails 的第 1、2 批，清洗后合计 5,058,910 条请求。每条请求的 total_tokens 等于 Request tokens 加 Response tokens，再聚合到左闭右开的 5 分钟窗口。时间网格补齐后共有 34,848 个窗口，零请求窗口保留为零。切分不是随机抽样，而是按目标时间连续切成 70%、15%、15%。固定突发阈值只由训练集确定，P95 为 315,246.2 Token/5min。第 3 批从主实验完全隔离，只做冻结模型的零样本稳健性检验。",
  "【约 1:35】这是全研究最关键的时间契约。窗口时间戳是左边界，因此在预测原点 t，t 这个 5 分钟窗尚未完整结束，不能当输入；最后可用的完整窗口是 t 减 5 分钟，目标是 t 加 15 分钟。训练、验证、测试的归属按目标时间决定，避免一个输入落在训练期、目标却跨到验证或测试期。所有滚动特征只使用历史，P95 和 scaler 只在训练范围拟合，超参数只看验证 MAE。我们还写了独立校验脚本，逐项检查这些不等式与拟合边界，干净复跑时全部通过。",
  "【约 1:20】比较对象包括两个透明基线和两个学习模型。Persistence 使用最近一个完整窗口；Seasonal Naive 使用历史相对周期。XGBoost 输入 36 个历史特征，包括 Token 滞后、滚动统计、请求量、平均输入输出长度、服务构成和相对周期相位；12 个候选只用验证 MAE 选择。LSTM 输入过去 144 个窗口，也就是 12 小时的 10 变量序列；网络只有一层 32 单元，共 5,537 个参数。选参冻结后，两种学习模型在训练加验证范围重拟合，测试只评估一次。",
  "【约 1:30】先看总体点预测。15 分钟主测试中，LSTM 的 MAE 是 11,054.5 Token/5min，Persistence 是 12,342.4，Seasonal Naive 是 15,770.3，XGBoost 是 16,464.4。因此 LSTM 相对 Persistence 改善约 10.4%，是这个时期的 MAE 最优。可能原因包括短期序列连续性和小型网络对平滑动态的吸收；但不能因此说深度模型在所有指标都最好。XGBoost 的特征工程可能更适合排序极端点，而测试时期的分布也明显不同。本页只得出点预测结论。",
  "【约 1:35】再看固定容量突发。测试集按训练期 P95 定义后只剩 5 个正例，四个模型都没有命中，所以 Precision、Recall 和 F1 都是零。XGBoost 的 AP 为 0.0113，略高于 LSTM 的 0.0108，说明排序能力存在很小差异；但这个量级离可部署检测仍很远。主要原因是测试期绝对负载下移，固定阈值落到了极端尾部，再加上正例过少，PR 曲线非常离散。零 F1 不等于模型毫无信息，但也不能用略高的 AP 宣称突发识别已经成功。",
  "【约 1:30】跨时期结果进一步说明模型排名不稳定。所有模型和阈值都冻结后直接应用到 BurstGPT 第 3 批，Persistence 的 MAE 是 43,838.2，F1 是 0.711，两个指标都最好；XGBoost 和 LSTM 的误差更高，F1 约为 0.51 和 0.52。这里没有使用第 3 批重新选参或拟合，因此它反映的是零样本迁移，而不是再次训练后的可达性能。可能原因包括负载水平、构成、周期和突发基率变化。结论是复杂模型的单时期领先不能替代跨时期验证。",
  "【约 1:20】所有数字都要和边界一起解释。主测试只有 5 个突发正例，使 F1 极不稳定；原始轨迹没有真实日期语义，因此相对日周相位不能解释为工作日或节假日；数据没有 request_id，删除全字段精确重复时可能把并发同值请求误判为重复；另外 LSTM 对负预测做了裁零，而 XGBoost 没有，处理并非完全对称。更重要的是，本研究只评价预测指标，没有直接测量 GPU 能耗、SLO 或调度收益，因此不能把相关性结果外推成基础设施收益。",
  "【约 1:15】最后总结。第一个答案：点预测上，LSTM 在主测试期的 MAE 最低。第二个答案：固定训练 P95 的突发识别并未成功，四个模型 F1 都为零，XGBoost 只在 AP 上略高。第三个答案：跨时期迁移时 Persistence 重新领先，说明模型排名依赖时期和基率。下一步应把点预测扩展为概率区间，研究自适应阈值和在线更新，并用多个批次做滚动回测。所有脚本、版本、随机种子、表图产物和论文生成命令都已写入 README，可从干净环境复跑。谢谢。",
];

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const projectRoot = path.resolve(args["project-root"] || path.resolve(import.meta.dirname, "..", ".."));
  const outDir = path.resolve(args["out-dir"] || path.join(projectRoot, "deliverables"));
  const previewDir = path.resolve(args["preview-dir"] || path.join(projectRoot, "tmp", "presentation_preview"));
  await fs.mkdir(outDir, { recursive: true });
  await fs.mkdir(previewDir, { recursive: true });

  const figRoot = path.join(projectRoot, "artifacts", "final", "figures");
  const [mainFig, burstFig, robustnessFig] = await Promise.all([
    imageBytes(path.join(figRoot, "fig_13_model_mae_f1_h15.png")),
    imageBytes(path.join(figRoot, "fig_12_burst_window_zoom_h15.png")),
    imageBytes(path.join(figRoot, "w5_python", "fig_w5_py_10_burstgpt3_robustness.png")),
  ]);

  const p = Presentation.create({ slideSize: SLIDE });
  addCover(p, notes[0]);
  addQuestion(p, notes[1]);
  addData(p, notes[2]);
  addLeakage(p, notes[3]);
  addModels(p, notes[4]);
  addMainResult(p, notes[5], mainFig);
  addBurst(p, notes[6], burstFig);
  addRobustness(p, notes[7], robustnessFig);
  addLimits(p, notes[8]);
  addConclusion(p, notes[9]);

  for (const [index, slide] of p.slides.items.entries()) {
    const stem = `slide-${String(index + 1).padStart(2, "0")}`;
    const png = await p.export({ slide, format: "png", scale: 1 });
    await fs.writeFile(path.join(previewDir, `${stem}.png`), new Uint8Array(await png.arrayBuffer()));
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(previewDir, `${stem}.layout.json`), await layout.text(), "utf8");
  }
  const montage = await p.export({ format: "webp", montage: true, scale: 1 });
  await fs.writeFile(path.join(previewDir, "deck-montage.webp"), new Uint8Array(await montage.arrayBuffer()));

  const pptxPath = path.join(outDir, "LLM_Token_Load_Forecasting_15min.pptx");
  const pptx = await PresentationFile.exportPptx(p);
  await pptx.save(pptxPath);

  const notesPath = path.join(outDir, "LLM_Token_Load_Forecasting_15min_speaker_notes.md");
  const notesMd = [
    "# LLM Token 负载预测与突发识别：15 分钟讲稿",
    "",
    "目标时长：约 13 分 50 秒，另留 1 分 10 秒用于换页、停顿与现场节奏。",
    "",
    ...notes.flatMap((n, i) => [`## 第 ${i + 1} 页`, "", n, ""]),
  ].join("\n");
  await fs.writeFile(notesPath, notesMd, "utf8");
  console.log(JSON.stringify({ pptxPath, notesPath, previewDir, slides: p.slides.items.length }, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
