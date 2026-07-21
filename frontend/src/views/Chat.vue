<template>

<MainLayout>

<div class="chat-page">

<!-- 聊天窗口 -->
<div class="chat-box">

<div
v-for="(msg,index) in messages"
:key="index"
class="msg"
:class="msg.role"
>

<div class="bubble">

<div class="role">
{{ msg.role }}
</div>

<div class="content">
{{ msg.displayContent || msg.content }}
</div>

<div
v-if="msg.sources"
class="sources"
>

来源：

<div
v-for="(s,i) in msg.sources"
:key="i"
>

📄 {{ s.source }}
P{{ s.page }}
Chunk{{ s.chunk_id }}

</div>

</div>

<div
v-if="msg.role === 'ai'"
class="actions"
>

<el-button
size="small"
@click="retry(msg)"
:loading="msg.loading"
>

重新生成

</el-button>

</div>

</div>

</div>

</div>

<!-- 输入区 -->
<div class="input-box">

<el-button
@click="exportPDF"
icon="Download"
:loading="exporting"
:disabled="exporting || messages.length === 0"
>

导出PDF

</el-button>

<el-input
v-model="question"
placeholder="请输入问题..."
/>

<el-button
class="pixel-btn"
@click="send"
:loading="loading"
>

发送

</el-button>

</div>

</div>

</MainLayout>

</template>

<script setup>

import { ref,onMounted } from "vue";

import { ElMessage } from "element-plus";

import MainLayout
from "../layouts/MainLayout.vue";

import { askQuestion, askQuestionStream }
from "../api/chat";

import { getChatHistory }
from "../api/chat_history";

const question = ref("");

const loading = ref(false);
const exporting = ref(false);

const messages = ref([]);

// 每次进入页面生成一个会话ID，同一页面内的追问属于同一 session（多轮对话）
const sessionId = ref(
  (crypto.randomUUID && crypto.randomUUID()) ||
  `s_${Date.now()}_${Math.random().toString(36).slice(2)}`
);

const formatExportFilename = () => {
  const now = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}`;
};

const escapeHtml = (text) => {
  return String(text ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
};

const buildExportHtml = () => {
  const exportTime = new Date().toLocaleString("zh-CN");
  let questionNo = 0;
  let blocks = "";

  messages.value.forEach((msg) => {
    if (msg.role === "user") {
      questionNo += 1;
      blocks += `
        <div class="block user-block">
          <div class="label">问题 ${questionNo}</div>
          <div class="text">${escapeHtml(msg.content)}</div>
        </div>`;
    } else {
      let sourcesHtml = "";
      if (msg.sources?.length) {
        const items = msg.sources
          .map(
            (s, i) =>
              `<li>${i + 1}. ${escapeHtml(s.source)} · P${s.page} · Chunk${s.chunk_id}</li>`
          )
          .join("");
        sourcesHtml = `<div class="sources"><div class="label">参考来源</div><ul>${items}</ul></div>`;
      }
      blocks += `
        <div class="block ai-block">
          <div class="label">回答</div>
          <div class="text">${escapeHtml(msg.content)}</div>
          ${sourcesHtml}
        </div>`;
    }
  });

  return `
    <div xmlns="http://www.w3.org/1999/xhtml" style="
      width: 794px;
      padding: 32px;
      box-sizing: border-box;
      background: #fff;
      color: #1f2937;
      font-family: 'Microsoft YaHei', 'SimHei', sans-serif;
      font-size: 14px;
      line-height: 1.75;
    ">
      <style>
        .title { font-size: 22px; font-weight: bold; margin: 0 0 8px; color: #111827; }
        .meta { font-size: 12px; color: #6b7280; margin-bottom: 24px; }
        .block { margin-bottom: 20px; padding: 16px; border: 1px solid #d1d5db; border-radius: 8px; }
        .user-block { background: #eff6ff; border-color: #93c5fd; }
        .ai-block { background: #f9fafb; }
        .label { font-size: 12px; font-weight: bold; color: #374151; margin-bottom: 8px; }
        .text { white-space: pre-wrap; word-break: break-word; }
        .sources { margin-top: 12px; padding-top: 12px; border-top: 1px dashed #d1d5db; }
        .sources ul { margin: 6px 0 0; padding-left: 18px; }
        .sources li { margin: 4px 0; font-size: 12px; color: #4b5563; }
      </style>
      <div class="title">RAG 对话记录</div>
      <div class="meta">导出时间：${escapeHtml(exportTime)} · 共 ${questionNo} 组问答</div>
      ${blocks}
    </div>`;
};

const addCanvasToPdf = (pdf, canvas, pageWidth, pageHeight) => {
  const imgWidth = pageWidth;
  const pageCanvasHeightPx = Math.floor((pageHeight * canvas.width) / imgWidth);
  let offsetY = 0;
  let pageIndex = 0;

  while (offsetY < canvas.height) {
    if (pageIndex > 0) {
      pdf.addPage();
    }
    const sliceHeight = Math.min(pageCanvasHeightPx, canvas.height - offsetY);
    const pageCanvas = document.createElement("canvas");
    pageCanvas.width = canvas.width;
    pageCanvas.height = sliceHeight;
    const ctx = pageCanvas.getContext("2d");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, pageCanvas.width, pageCanvas.height);
    ctx.drawImage(
      canvas,
      0,
      offsetY,
      canvas.width,
      sliceHeight,
      0,
      0,
      canvas.width,
      sliceHeight
    );

    const imgData = pageCanvas.toDataURL("image/png");
    const renderHeight = (sliceHeight * imgWidth) / canvas.width;
    pdf.addImage(imgData, "PNG", 0, 0, imgWidth, renderHeight);

    offsetY += sliceHeight;
    pageIndex += 1;
  }
};

// 导出 PDF：离屏渲染完整对话文本，避免只截到可见区域
const exportPDF = async () => {
  if (messages.value.length === 0) {
    ElMessage.warning("暂无对话可导出");
    return;
  }

  exporting.value = true;
  let container = null;

  try {
    const { jsPDF } = await import("jspdf");
    const html2canvas = (await import("html2canvas")).default;

    container = document.createElement("div");
    container.style.cssText =
      "position: fixed; left: -10000px; top: 0; z-index: -1; pointer-events: none;";
    container.innerHTML = buildExportHtml();
    document.body.appendChild(container);

    const target = container.firstElementChild;
    const canvas = await html2canvas(target, {
      scale: 2,
      useCORS: true,
      logging: false,
      backgroundColor: "#ffffff",
      width: target.scrollWidth,
      height: target.scrollHeight,
      windowWidth: target.scrollWidth,
      windowHeight: target.scrollHeight,
    });

    const pdf = new jsPDF("p", "mm", "a4");
    const pageWidth = pdf.internal.pageSize.getWidth();
    const pageHeight = pdf.internal.pageSize.getHeight();
    addCanvasToPdf(pdf, canvas, pageWidth, pageHeight);

    pdf.save(`RAG对话记录_${formatExportFilename()}.pdf`);
    ElMessage.success("PDF 导出成功");
  } catch (e) {
    console.error(e);
    ElMessage.error("PDF 导出失败");
  } finally {
    if (container?.parentNode) {
      container.parentNode.removeChild(container);
    }
    exporting.value = false;
  }
};

// 流式请求封装：真实 SSE 渲染，失败时降级为非流式接口
const streamAnswer = async (msg, query) => {
  msg.content = "";
  msg.displayContent = "";
  msg.sources = null;

  let streamFailed = false;

  try {
    await askQuestionStream(query, sessionId.value, {
      onToken: (text) => {
        msg.content += text;
        msg.displayContent = msg.content;
      },
      onSources: (sources) => {
        msg.sources = sources;
      },
      onError: (err) => {
        streamFailed = true;
        console.warn("SSE 错误事件:", err);
      }
    });
  } catch (e) {
    streamFailed = true;
    console.warn("流式请求失败，尝试非流式降级:", e);
  }

  // 降级：流式失败且没有任何输出时走原非流式接口
  if (streamFailed && !msg.content) {
    const res = await askQuestion(query, sessionId.value);
    msg.content = res.data.answer;
    msg.displayContent = res.data.answer;
    msg.sources = res.data.sources;
  }
};

// 重新生成
const retry = async (msg) => {
  // 找到对应的用户问题
  const msgIndex = messages.value.indexOf(msg);
  if (msgIndex <= 0) return;

  const userMsg = messages.value[msgIndex - 1];
  if (userMsg.role !== 'user') return;

  // 标记加载状态
  msg.loading = true;

  try {
    await streamAnswer(msg, userMsg.content);
  } catch (e) {
    msg.content = "请求失败";
    msg.displayContent = "请求失败";
  }

  msg.loading = false;
};

const send = async () => {

if (!question.value) return;

messages.value.push({
role: "user",
content: question.value,
displayContent: question.value
});

const q = question.value;

question.value = "";

loading.value = true;

const aiMsg = {
role: "ai",
content: "",
sources: null,
displayContent: "",
loading: false
};

messages.value.push(aiMsg);

try {

await streamAnswer(aiMsg, q);

} catch (e) {

aiMsg.content = "请求失败";
aiMsg.displayContent = "请求失败";

}

loading.value = false;

};

const loadHistory = async () => {
  try {
    const res = await getChatHistory();
    const history = res.data.history;

    // 历史记录按时间正序排列
    history.reverse().forEach(item => {

      let parsedSources = [];

      try {
        parsedSources = item.sources
          ? JSON.parse(item.sources)
          : [];
      } catch (e) {
        console.warn(
          "sources解析失败:",
          item.sources
        );
        parsedSources = [];
      }

      messages.value.push({
        role: "user",
        content: item.question,
        displayContent: item.question
      });

      messages.value.push({
        role: "ai",
        content: item.answer,
        sources: parsedSources,
        displayContent: item.answer,
        loading: false
      });

    });
  } catch (e) {
    console.error("加载历史记录失败", e);
  }
};

onMounted(() => {
  loadHistory();
});

</script>

<style scoped>

.chat-page {

display: flex;

flex-direction: column;

height: 100%;

}

.chat-box {

flex: 1;

overflow-y: auto;

padding: 20px;

background: #F5E6C8;

}

.msg {

margin-bottom: 20px;

display: flex;

}

.msg.user {

justify-content: flex-end;

}

.msg.ai {

justify-content: flex-start;

}

.bubble {

max-width: 60%;

padding: 12px;

border: 4px solid #4E342E;

background: #FFF8E1;

box-shadow: 4px 4px 0 #3E2723;

}

.role {

font-size: 12px;

font-weight: bold;

margin-bottom: 6px;

}

.content {

white-space: pre-wrap;

}

.sources {

margin-top: 10px;

font-size: 12px;

color: #6D4C41;

border-top: 2px dashed #4E342E;

padding-top: 6px;

}

.sources div {

padding: 4px 0;

}

.actions {

margin-top: 10px;

display: flex;

gap: 10px;

}

.input-box {

display: flex;

gap: 10px;

padding: 10px;

border-top: 4px solid #4E342E;

background: #FFF8E1;

}

</style>