---
language:
- zh
pipeline_tag: text-generation
license: apache-2.0
task_categories:
- text-generation
size_categories:
- 10B<n<100B
---


# **Chinese Cosmopedia Dataset**          [[中文]](#chinese)    [[English]](#english)

<a id="english"></a>

<p align="center">
<img width="600px" alt="OpenCSG" src="./logo.jpg">
</p>


<p align="center"><a href="https://portal.opencsg.com/models">[OpenCSG Community]</a>   <a href="https://github.com/OpenCSGs/Awesome-SLMs">[github]</a>  <a href="https://cdn-uploads.huggingface.co/production/uploads/64c71b27d43e4dee51a8b31a/HU6vz21qKTEmUBCWqCFh9.jpeg">[wechat]</a>  <a href="https://twitter.com/OpenCsg">[Twitter]</a> </p>


</div>
The Chinese Cosmopedia dataset contains a total of 15 million entries, approximately 60B tokens. Two key elements in constructing the synthetic dataset are seed data and prompts. Seed data determines the theme of the generated content, while prompts define the style of the data (such as textbooks, stories, tutorials, or children's books). The data sources are diverse, including Chinese Wikipedia, Baidu Baike, Zhihu Q&A, and technical blogs, ensuring both the breadth and authority of the content. The generated data comes in various formats, such as university textbooks, middle school textbooks, children's stories, ordinary stories, and WikiHow-style tutorials. By generating multiple styles for each piece of seed data, the dataset is not only suitable for academic research but also widely applicable in education, entertainment, and technology fields.<br><br>




# Data Sources and Types

The Chinese Cosmopedia draws from a variety of rich data sources, covering a wide range of Chinese content platforms and knowledge bases, including:

- **Chinese Wikipedia**: Provides a large number of precise, authoritative knowledge articles.
- **Baidu Baike**: As one of the most influential encyclopedia platforms in China, Baidu Baike supplies an extensive range of Chinese knowledge resources for the dataset.
- **Zhihu Q&A**: Content extracted from the interactive Q&A platform, covering discussions and insights across various fields.
- **Technical Blogs**: Articles from technical communities, covering in-depth discussions on topics ranging from programming to artificial intelligence.

These seed data sources form the core content of the Chinese Cosmopedia dataset, ensuring coverage of knowledge across multiple domains.


# Data Formats and Styles

The Chinese Cosmopedia dataset places special emphasis on the style and format of the generated content, encompassing various text types from academic to practical applications. The main categories include:

- **University Textbooks**: Structured rigorously, delving into core concepts of various university disciplines.
- **Middle School Textbooks**: Educational content aimed at middle school students, concise and easy to understand, focusing on basic knowledge transmission.
- **Children’s Stories**: Designed for 5-year-old children, using simple language to help young kids understand the world and interpersonal relationships.
- **Ordinary Stories**: Employing engaging plots and character dialogues to vividly illustrate specific concepts.
- **WikiHow-style Tutorials**: Detailed step-by-step guides to help users accomplish specific tasks.

Each text type has been carefully adjusted in style based on its application scenario and target audience. Through this design, Cosmopedia can be applied not only to academic research but also broadly in fields such as education, entertainment, and technology.

# Statistics

Seed Data Sources: {'blog': 2,111,009, 'baike': 10,939,121, 'wiki': 173,671, 'knowledge QA': 2,291,547}

<p align="center">
<img width="900px" alt="OpenCSG" src="./datasource.png">
</p>

Data Formats: {'preschool story': 1,637,760, 'normal story': 3,332,288, 'middle school textbook': 4,677,397, 'college textbook': 3,127,902, 'wikihow': 2,740,001}

<p align="center">
<img width="600px" alt="OpenCSG" src="./datastyle.png">
</p>




# Data Generation and Model

The Chinese Cosmopedia’s data generation is based on the OpenCSG team's proprietary OpenCSG-Wukong-Enterprise-Long model. This model, with its robust long-text generation capability, ensures the coherence and depth of the generated content. During the data generation process, the OpenCSG team designed specific prompts for each text type and content category to ensure that the generated data's style matches its intended format. For example, prompts for textbook content guide the model to generate rigorous and in-depth academic texts, while prompts for story content inspire the creation of vivid and engaging narratives.

Prompts Used for Generating Various Data Formats:

University Textbook

```Plain
这是一段来自网页的摘录：“{}”。
请编写一个针对大学生的足够详细的教科书课程单元，该单元与给定的摘录中的某个概念或多个概念相关。
不需要包含摘录中的所有内容，只需要发掘其中适合作为教科书内容的部分。你可以自由补充其他相关知识。
不能仅仅列出概念，而是要深入发展和详细探讨每个概念，因为我们优先考虑深入理解主题内容，而不是广度。
要求：1. 严谨性：确保对概念/章节的深入覆盖。
2. 吸引性：用学术、专业且引人入胜的语气撰写，以吸引兴趣。
3. 应用：融入具体的实践例子，例如微积分中要给出公式、严格证明，历史中要给出关键日期和人物，计算机操作中要给出代码。
4.不需要给出参考文献。内容中不应包含广告或涉及隐私的信息。
请记住，要针对大学生制作内容，他们可能拥有一些基础知识，但不是该领域的专家。内容应该详细且发人深省。
请立即开始撰写教科书，不要使用图片，不要输出除了教科书以外的内容。
```

Middle School Textbook

```Plain
网页摘录：“{}”。
创建一个与上述网页摘录中的某个概念相关的具有教育意义的内容，针对中学生，尽量长而详细。你可以自由补充其他相关知识。
不能仅仅列出概念，而是要深入发展和详细探讨每个概念，因为我们优先考虑深入理解主题内容，而不是广度，不需要包含摘录中的所有内容。
不应该使用像微积分这样的复杂大学级主题，因为这些通常不是中学的内容。
如果主题是关于这些的，寻找一个更简单的科学替代内容来解释，并使用日常例子。
例如，如果主题是“线性代数”，你可能会讨论如何通过将物体排列成行和列来解决谜题。
避免使用技术术语和LaTeX，只讨论中学级别的主题。内容中不应包含广告或涉及隐私的信息。
请直接开始撰写教育内容，不要输出除了教育内容以外的内容。
```

Nomal Story

```Plain
写一个与以下文本片段相关的引人入胜的故事：“{}”。
故事不需要提及片段中的所有内容，只需使用它来获得灵感并发挥创意！可以加入其它知识。
故事应包括： 1.小众概念或兴趣：深入研究特定的概念、爱好、兴趣或幽默情况 
2.意想不到的情节转折或引人入胜的冲突，引入具有挑战性的情况或困境。 
3.对话：故事必须至少包含一个有意义的对话，以揭示人物深度、推进情节或揭开谜团的关键部分
4.反思和洞察：以具有教育意义的新理解、启示的结论结束。 
5.故事中的人物应使用中国式的名字。请勿包含广告或涉及隐私的信息。
请马上开始讲故事，不要输出除了故事以外的内容。
```

Preschool Story

```Plain
网页摘录：“{}”
创建一个与上述网页摘录中的某个概念相关的具有教育意义的儿童故事，重点针对对世界和人际交往零知识的5岁儿童。
故事不需要提及片段中的所有内容，只需使用它来获得灵感并发挥创意。
故事应该使用简单的术语。你可以补充额外的知识来帮助理解。
使用易于理解的示例，并将 5 岁儿童可能提出的问题及其答案纳入故事中。故事应涵盖日常行为和常见物品的使用。
不应该使用像微积分这样的复杂大学级主题，因为这些通常不是幼儿能理解的内容。如果主题是关于这些的，寻找一个更简单的科学替代内容来解释，并使用日常例子。例如，如果主题是“线性代数”，你可能会讨论如何通过将物体排列成行和列来解决谜题。
请直接开始撰写故事，不要输出除了故事以外的内容。
```

wikihow

```Plain
网页摘录：“{}”。
以 WikiHow 的风格写一篇长而非常详细的教程，教程与此网页摘录有相关性。
教程中需要包括对每个步骤的深入解释以及它如何帮助实现预期结果。你可以自由补充其他相关知识。
确保清晰性和实用性，让读者能够轻松遵循教程完成任务。内容中不应包含广告或涉及隐私的信息。
不要使用图像。请直接开始撰写教程。
```

## License Agreement

Usage of the Chinese Cosmopedia dataset requires adherence to the OpenCSG Community License. The Chinese Cosmopedia dataset supports commercial use. If you plan to use the OpenCSG model or its derivatives for commercial purposes, you must comply with the terms and conditions outlined in the OpenCSG Community License as well as the Apache 2.0 License. For commercial use, please send an email to lorraineg@opencsg.com and obtain permission.

**We warmly invite developers and researchers interested in this field to follow and engage with the community, working together to advance the technology. Stay tuned for the open-source release of the dataset!**

<a id="chinese"></a>

<p>
</p>

# Chinese Cosmopedia 数据集介绍

<p align="center">
<img width="600px" alt="OpenCSG" src="./logo.jpg">
</p>



<p align="center"><a href="https://opencsg.com/models">[OpenCSG 社区]</a>   <a href="https://github.com/OpenCSGs/Awesome-SLMs">[github]</a>  <a href="https://cdn-uploads.huggingface.co/production/uploads/64c71b27d43e4dee51a8b31a/HU6vz21qKTEmUBCWqCFh9.jpeg">[微信]</a>  <a href="https://twitter.com/OpenCsg">[推特]</a> </p>



</div>
Chinese Cosmopedia数据集共包含1500万条数据，约60B个token，构建合成数据集的两个核心要素是种子数据和prompt。种子数据决定了生成内容的主题，prompt则决定了数据的风格（如教科书、故事、教程或幼儿读物）。数据来源丰富多样，涵盖了中文维基百科、百度百科、知乎问答和技术博客等平台，确保内容的广泛性和权威性。生成的数据形式多样，涵盖大学教科书、中学教科书、幼儿故事、普通故事和WikiHow风格教程等多种不同风格。通过对每条种子数据生成多种不同风格的内容，数据集不仅适用于学术研究，还广泛应用于教育、娱乐和技术领域。<br><br>



# 数据来源与种类

Chinese Cosmopedia的数据来源丰富，涵盖了多种中文内容平台和知识库，包括：

- **中文维基百科**：提供了大量精确、权威的知识性文章。
- **百度百科**：作为国内最具影响力的百科平台之一，百度百科为数据集提供了广泛的中文知识资源。
- **知乎问答**：从互动式问答平台中提取的内容，涵盖了多个领域的讨论与见解。
- **技术博客**：来自技术社区的文章，涵盖了从编程到人工智能等多个技术方向的深入讨论。

这些种子数据构成了Chinese Cosmopedia数据集的核心内容来源，确保了不同领域知识的覆盖。


# 数据形式与风格

Chinese Cosmopedia数据集特别注重生成内容的风格与形式，涵盖了从学术到日常应用的多种文本类型，主要包括以下几类：

- **大学教科书**：内容结构严谨，深入探讨各类大学学科的核心概念。
- **中学教科书**：适合中学生的教学内容，简洁易懂，注重基本知识的传达。
- **幼儿故事**：面向5岁儿童，语言简洁易懂，帮助幼儿理解世界和人际关系。
- **普通故事**：通过引人入胜的情节和人物对话，展开对某一概念的生动描述。
- **WikiHow风格教程**：详细的步骤指导，帮助用户完成特定任务。

每种文体都根据不同的应用场景和目标读者群体，进行了精细化的风格调整。通过这种设计，Cosmopedia不仅适用于学术研究，还能广泛应用于教育、娱乐、技术等领域。

# 统计

种子数据来源：{'blog': 2,111,009, 'baike': 10,939,121, 'wiki': 173,671, 'knowledge QA': 2,291,547}

<p align="center">
<img width="900px" alt="OpenCSG" src="./datasource.png">
</p>

数据形式： {'preschool story': 1,637,760, 'normal story': 3,332,288, 'middle school textbook': 4,677,397, 'college textbook': 3,127,902, 'wikihow': 2,740,001}

<p align="center">
<img width="600px" alt="OpenCSG" src="./datastyle.png">
</p>



# 数据生成与模型

Chinese Cosmopedia的数据生成基于OpenCSG团队自主开发的**OpenCSG-Wukong-Enterprise-Long**模型。该模型通过强大的长文本生成能力，确保了生成数据的连贯性和内容深度。在数据生成过程中，OpenCSG团队为每种文体和内容类型设计了专门的prompt（提示词），以确保数据生成的风格与内容准确匹配。例如，对于教科书类型的内容，prompt会引导模型生成严谨且具有深度的学术文本，而对于故事类内容，则引导模型创造生动、引人入胜的情节。

我们用于生成各种格式的数据的prompt如下

大学教科书

```Plain
这是一段来自网页的摘录：“{}”。
请编写一个针对大学生的足够详细的教科书课程单元，该单元与给定的摘录中的某个概念或多个概念相关。
不需要包含摘录中的所有内容，只需要发掘其中适合作为教科书内容的部分。你可以自由补充其他相关知识。
不能仅仅列出概念，而是要深入发展和详细探讨每个概念，因为我们优先考虑深入理解主题内容，而不是广度。
要求：1. 严谨性：确保对概念/章节的深入覆盖。
2. 吸引性：用学术、专业且引人入胜的语气撰写，以吸引兴趣。
3. 应用：融入具体的实践例子，例如微积分中要给出公式、严格证明，历史中要给出关键日期和人物，计算机操作中要给出代码。
4.不需要给出参考文献。内容中不应包含广告或涉及隐私的信息。
请记住，要针对大学生制作内容，他们可能拥有一些基础知识，但不是该领域的专家。内容应该详细且发人深省。
请立即开始撰写教科书，不要使用图片，不要输出除了教科书以外的内容。
```

中学教科书

```Plain
网页摘录：“{}”。
创建一个与上述网页摘录中的某个概念相关的具有教育意义的内容，针对中学生，尽量长而详细。你可以自由补充其他相关知识。
不能仅仅列出概念，而是要深入发展和详细探讨每个概念，因为我们优先考虑深入理解主题内容，而不是广度，不需要包含摘录中的所有内容。
不应该使用像微积分这样的复杂大学级主题，因为这些通常不是中学的内容。
如果主题是关于这些的，寻找一个更简单的科学替代内容来解释，并使用日常例子。
例如，如果主题是“线性代数”，你可能会讨论如何通过将物体排列成行和列来解决谜题。
避免使用技术术语和LaTeX，只讨论中学级别的主题。内容中不应包含广告或涉及隐私的信息。
请直接开始撰写教育内容，不要输出除了教育内容以外的内容。
```

普通故事

```Plain
写一个与以下文本片段相关的引人入胜的故事：“{}”。
故事不需要提及片段中的所有内容，只需使用它来获得灵感并发挥创意！可以加入其它知识。
故事应包括： 1.小众概念或兴趣：深入研究特定的概念、爱好、兴趣或幽默情况 
2.意想不到的情节转折或引人入胜的冲突，引入具有挑战性的情况或困境。 
3.对话：故事必须至少包含一个有意义的对话，以揭示人物深度、推进情节或揭开谜团的关键部分
4.反思和洞察：以具有教育意义的新理解、启示的结论结束。 
5.故事中的人物应使用中国式的名字。请勿包含广告或涉及隐私的信息。
请马上开始讲故事，不要输出除了故事以外的内容。
```

幼儿故事

```Plain
网页摘录：“{}”
创建一个与上述网页摘录中的某个概念相关的具有教育意义的儿童故事，重点针对对世界和人际交往零知识的5岁儿童。
故事不需要提及片段中的所有内容，只需使用它来获得灵感并发挥创意。
故事应该使用简单的术语。你可以补充额外的知识来帮助理解。
使用易于理解的示例，并将 5 岁儿童可能提出的问题及其答案纳入故事中。故事应涵盖日常行为和常见物品的使用。
不应该使用像微积分这样的复杂大学级主题，因为这些通常不是幼儿能理解的内容。如果主题是关于这些的，寻找一个更简单的科学替代内容来解释，并使用日常例子。例如，如果主题是“线性代数”，你可能会讨论如何通过将物体排列成行和列来解决谜题。
请直接开始撰写故事，不要输出除了故事以外的内容。
```

wikihow教程

```Plain
网页摘录：“{}”。
以 WikiHow 的风格写一篇长而非常详细的教程，教程与此网页摘录有相关性。
教程中需要包括对每个步骤的深入解释以及它如何帮助实现预期结果。你可以自由补充其他相关知识。
确保清晰性和实用性，让读者能够轻松遵循教程完成任务。内容中不应包含广告或涉及隐私的信息。
不要使用图像。请直接开始撰写教程。
```


## 许可协议
使用 Chinese Cosmopedia 数据集需要遵循 OpenCSG 社区许可证。Chinese Cosmopedia 数据集支持商业用途。如果您计划将 OpenCSG 模型或其衍生产品用于商业目的，您必须遵守 OpenCSG 社区许可证以及 Apache 2.0 许可证中的条款和条件。如用于商业用途，需发送邮件至 lorraineg@opencsg.com，并获得许可。

**我们诚邀对这一领域感兴趣的开发者和研究者关注和联系社区，共同推动技术的进步。敬请期待数据集的开源发布！**
