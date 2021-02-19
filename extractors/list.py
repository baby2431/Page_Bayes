import math
import operator
from loguru import logger
import numpy as np
import re
from collections import defaultdict
from urllib.parse import urljoin
from gerapy_auto_extractor.utils.cluster import cluster_dict
from gerapy_auto_extractor.utils.preprocess import preprocess4list_extractor
from gerapy_auto_extractor.extractors.base import BaseExtractor
from gerapy_auto_extractor.utils.element import descendants_of_body
from gerapy_auto_extractor.schemas.element import Element
from gerapy_auto_extractor.patterns.datetime import METAS_CONTENT, REGEXES as DATE_REGEXES

LIST_MIN_NUMBER = 5
LIST_MIN_LENGTH = 8
LIST_MAX_LENGTH = 44
SIMILARITY_THRESHOLD = 0.8


class ListExtractor(BaseExtractor):
    """
    extract list from index page
    """

    def __init__(self, min_number=LIST_MIN_NUMBER, min_length=LIST_MIN_LENGTH, max_length=LIST_MAX_LENGTH,
                 similarity_threshold=SIMILARITY_THRESHOLD):
        """
        init list extractor
        """
        super(ListExtractor, self).__init__()
        self.min_number = min_number
        self.min_length = min_length
        self.max_length = max_length
        self.avg_length = (self.min_length + self.max_length) / 2
        self.similarity_threshold = similarity_threshold

    def _probability_of_title_with_length(self, length):
        """
        get the probability of title according to length
        import matplotlib.pyplot as plt
        x = np.asarray(range(5, 40))
        y = list_extractor.probability_of_title_with_length(x)
        plt.plot(x, y, 'g', label='m=0, sig=2')
        plt.show()
        :param length:
        :return:
        """
        sigma = 6
        return np.exp(-1 * ((length - self.avg_length) ** 2) / (2 * (sigma ** 2))) / (math.sqrt(2 * np.pi) * sigma)

    def _build_clusters(self, element):
        """
        内容元素判断是否像一个列表节点
        build candidate clusters according to element
        :return:
        """
        descendants_tree = defaultdict(list)
        descendants = descendants_of_body(element)
        for descendant in descendants:
            # if one element does not have enough siblings, it can not become a child of candidate element //他要是没有兄弟元素，
            if descendant.number_of_siblings + 1 < self.min_number:
                continue
            # if min length is larger than specified max length, it can not become a child of candidate element
            if descendant.a_descendants_group_text_min_length > self.max_length:
                continue
            # if max length is smaller than specified min length, it can not become a child of candidate element
            if descendant.a_descendants_group_text_max_length < self.min_length:
                continue
            # descendant element must have same siblings which their similarity should not below similarity_threshold
            if descendant.similarity_with_siblings < self.similarity_threshold:
                continue
            descendants_tree[descendant.parent_selector].append(descendant)
        descendants_tree = dict(descendants_tree)
        'html>body>div[class="within_bcrightfr"]:nth-child(1)>div[class="within_bcrtxtwo"]:nth-child(2)'
        # cut tree, remove parent block
        selectors = sorted(list(descendants_tree.keys()))
        last_selector = None
        # 这里有点问题。。。多个列表识别不了
        for selector in selectors[::-1]:
            # if later selector
            if last_selector and selector and last_selector.startswith(selector):
                del descendants_tree[selector]
            last_selector = selector
        clusters = cluster_dict(descendants_tree)
        return clusters

    def _evaluate_cluster(self, cluster):
        """
        calculate score of cluster using similarity, numbers, or other info
        :param cluster:
        :return:
        """
        score = dict()

        # calculate avg_similarity_with_siblings
        score['avg_similarity_with_siblings'] = np.mean(
            [element.similarity_with_siblings for element in cluster])

        # calculate number of elements
        score['number_of_elements'] = len(cluster)

        # calculate probability of it contains title
        # score['probability_of_title_with_length'] = np.mean([
        #     self._probability_of_title_with_length(len(a_descendant.text)) \
        #     for a_descendant in itertools.chain(*[element.a_descendants for element in cluster]) \
        #     ])

        # TODO: add more quota to select best cluster
        score['clusters_score'] = \
            score['avg_similarity_with_siblings'] \
            * np.log10(score['number_of_elements'] + 1) \
            # * clusters_score[cluster_id]['probability_of_title_with_length']
        return score

    def _extend_cluster(self, cluster):
        """
        extend cluster's elements except for missed children
        :param cluster:
        :return:
        """
        result = [element.selector for element in cluster]
        for element in cluster:
            path_raw = element.path_raw
            siblings = list(element.siblings)
            for sibling in siblings:
                # skip invalid element
                if not isinstance(sibling, Element):
                    continue
                sibling_selector = sibling.selector
                sibling_path_raw = sibling.path_raw
                if sibling_path_raw != path_raw:
                    continue
                # add missed sibling
                if sibling_selector not in result:
                    cluster.append(sibling)
                    result.append(sibling_selector)

        cluster = sorted(cluster, key=lambda x: x.nth)
        logger.log('inspect', f'cluster after extend {cluster}')
        return cluster

    def _best_cluster(self, clusters):
        """
        use clustering algorithm to choose best cluster from candidate clusters
        :param clusters:
        :return:
        """
        if not clusters:
            logger.log('inspect', 'there is on cluster, just return empty result')
            return []
        if len(clusters) == 1:
            logger.log('inspect', 'there is only one cluster, just return first cluster')
            return clusters[0]
        # choose best cluster using score
        clusters_score = defaultdict(dict)
        clusters_score_arg_max = 0
        clusters_score_max = -1
        for cluster_id, cluster in clusters.items():
            # calculate avg_similarity_with_siblings
            clusters_score[cluster_id] = self._evaluate_cluster(cluster)
            # get max score arg index
            if clusters_score[cluster_id]['clusters_score'] > clusters_score_max:
                clusters_score_max = clusters_score[cluster_id]['clusters_score']
                clusters_score_arg_max = cluster_id
        logger.log('inspect', f'clusters_score {clusters_score}')
        best_cluster = clusters[clusters_score_arg_max]
        return best_cluster

    def _extract_cluster(self, cluster):
        """
        extract title and href from best cluster
        :param cluster:
        :return:
        """
        if not cluster:
            return None
        # get best tag path of title
        probabilities_of_title = defaultdict(list)
        for element in cluster:
            descendants = element.a_descendants  # element 是div标签
            # descendants 是a标签 a 标签里面找到所有元素 判断是否是文字 还是 日期

            for descendant in descendants:
                path = descendant.path
                descendant_text = descendant.text
                probability_of_title_with_length = self._probability_of_title_with_length(len(descendant_text))
                # probability_of_title_with_descendants = self.probability_of_title_with_descendants(descendant)
                # TODO: add more quota to calculate probability_of_title
                probability_of_title = probability_of_title_with_length
                probabilities_of_title[path].append(probability_of_title)

        # get most probable tag_path
        probabilities_of_title_avg = {k: np.mean(v) for k, v in probabilities_of_title.items()}
        if not probabilities_of_title_avg:
            return None
        best_path = max(probabilities_of_title_avg.items(), key=operator.itemgetter(1))[0]
        logger.log('inspect', f'best tag path {best_path}')
        # 在这里得到需要的信息 #### 重点
        # extract according to best tag path
        result = []
        for element in cluster:
            descendants = element.a_descendants
            for descendant in descendants:
                path = descendant.path
                if path != best_path:
                    continue
                title = descendant.text
                url = descendant.attrib.get('href')
                if not url:
                    continue
                if url.startswith('//'):
                    url = 'http:' + url
                base_url = self.kwargs.get('base_url')
                if base_url:
                    url = urljoin(base_url, url)
                date, date_path = self._get_date_key(element)
                titles = self._get_title_key(descendant, date_path)
                result.append({
                    'title': title,
                    'titles': titles,
                    'date_path': "" if date_path is None else "/"+date_path.path_nth,
                    'date': date,
                    'url': url,
                    'url_path': "" if descendant is None else "/"+descendant.path_nth,
                })
        return result

    def _get_date_key(self, element: Element):
        """
        寻找元素里面的日期内容
        :params element:父级元素
        :return: 目标日期对象
        """
        date = ""
        date_path = None
        for child in element.children:
            flag = False
            if child.children:
                date, date_path = self._get_date_key(child)
                if date_path is not None:
                    break
            else:
                text = ''.join(child.xpath('.//text()'))
                if not text:
                    continue
                for regex in DATE_REGEXES:
                    result = re.search(regex, text)
                    if result:
                        date = result.group(1)
                        date_path = child
                        flag = True
                        break
            if flag:
                break

        return date, date_path

    def _get_title_key(self, element: Element, list_element):
        """
        寻找元素里面的类似标题元素
        实现思路，直接列表里找到最后的元素 ，判断是否有 包含文字的内容 最后和date元素的内容要去除
        :params element:父级元素
        :return: 目标标题
        """

        text = ''.join(element.xpath('.//text()'))
        titles = []
        # 如果没有文本 就没有必要往下了
        if not text:
            return []
        # 如果里面搜索到的是 非字符型数据。拉倒
        result = re.compile(u'[\u4e00-\u9fa5]')
        if not result.search(text):
            return []
        if not element.children:
            titles.append({'title': text, 'path': "/"+element.path_nth})
        for child in element.children:
            text = ''.join(child.xpath('.//text()'))
            if not text:
                continue
            # 字符判断
            result = re.compile(u'[\u4e00-\u9fa5]')
            if not result.search(text):
                continue
            # 如果还有子元素 直接到达子元素
            if child.children:
                titles.append(self._get_title_key(child, list_element))
            else:
                if child == list_element:
                    continue
                titles.append({'title': text, 'path': "/"+child.path_nth})
        return titles

    def process(self, element: Element):
        """
        extract content from html
        :param element:
        :return:
        """
        # preprocess
        preprocess4list_extractor(element)

        # build clusters
        clusters = self._build_clusters(element)
        logger.log('inspect', f'after build clusters {clusters}')

        # choose best cluster
        best_cluster = self._best_cluster(clusters)
        logger.log('inspect', f'best cluster {best_cluster}')

        extended_cluster = self._extend_cluster(best_cluster)
        logger.log('inspect', f'extended cluster {extended_cluster}')

        # extract result from best cluster
        return self._extract_cluster(best_cluster)


list_extractor = ListExtractor()


def extract_list(html, **kwargs):
    """
    提取
    extract list from index html
    :param: base_url
    :return:
    """
    return list_extractor.extract(html, **kwargs)
