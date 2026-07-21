<template>

<MainLayout>

<div class="pixel-card">

<h2>
知识库参数配置
</h2>

<el-form>

<el-form-item
label="Chunk Size"
>

<el-slider

v-model="config.chunk_size"

:min="100"

:max="2000"

/>

</el-form-item>

<el-form-item
label="Chunk Overlap"
>

<el-slider

v-model="config.chunk_overlap"

:min="0"

:max="500"

/>

</el-form-item>

<el-form-item
label="切分策略"
>

<el-radio-group
v-model="config.split_mode"
>

<el-radio
label="zh"
>
中文智能
</el-radio>

<el-radio
label="paragraph"
>
段落切分
</el-radio>

<el-radio
label="sentence"
>
句子切分
</el-radio>

</el-radio-group>

</el-form-item>

<el-form-item
label="Top K"
>

<el-input-number

v-model="config.top_k"

:min="1"

:max="10"

/>

</el-form-item>

</el-form>

<div
v-if="buildStatus.status === 'building'"
class="progress-area"
>

<p>正在构建向量库...{{ buildStatus.stage ? `（${buildStatus.stage}）` : "" }}</p>

<el-progress
:percentage="buildStatus.progress"
:status="buildStatus.status === 'completed' ? 'success' : ''"
/>

</div>

<el-button

class="pixel-btn"

@click="save"

:loading="buildStatus.status === 'building'"
>

重建知识库

</el-button>

</div>

</MainLayout>

</template>

<script setup>

import { reactive,onMounted,ref } 
from "vue";

import { ElMessage } 
from "element-plus";

import MainLayout
from "../layouts/MainLayout.vue";

import {

getConfig,
buildDb,
getBuildStatus

}

from "../api/rag";

const config =
reactive({

chunk_size:500,

chunk_overlap:100,

split_mode:"zh",

top_k:5

});

const buildStatus = ref({
progress: 0,
status: "idle"
});

let progressTimer = null;

const checkProgress = async () => {
  try {
    const res = await getBuildStatus();
    buildStatus.value = res.data;

    if (buildStatus.value.status === 'building') {
      progressTimer = setTimeout(checkProgress, 1000);
    } else if (buildStatus.value.status === 'completed') {
      ElMessage.success("知识库重建完成");
    } else if (buildStatus.value.status === 'failed') {
      ElMessage.error(buildStatus.value.error || "知识库重建失败");
    }
  } catch (e) {
    console.error("获取进度失败", e);
  }
};

onMounted(
async()=>{

const res =
await getConfig();

Object.assign(
config,
res.data
);

// 检查是否有正在进行的构建
checkProgress();
}
);

const save =
async()=>{

// 提交后台构建任务，立即返回 task_id，随后轮询真实进度
const res = await buildDb(
config
);

buildStatus.value = { progress: 0, status: "building", stage: "任务已提交" };

ElMessage.info(
"构建任务已提交，正在后台执行..."
);

checkProgress();

};

</script>

<style scoped>

.progress-area {

margin: 20px 0;

padding: 15px;

background: var(--card-color);

border-radius: 8px;

}

.progress-area p {

margin-bottom: 10px;

font-weight: bold;

}

</style>