#include <iostream>
#include <vector>
#include <string>
#include <stack>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <map>
#include <set>
#include <queue>
#include <tuple>
#include <chrono>
#include <thread>
#include <mutex>
#include <algorithm>
#include <list>
#include <cmath>
#include <cstring>

// 每个数字都应该在数字所对应的位置上面
int find_repeated_num(std::vector<int>& nums) {
    std::vector<int> postions(nums.size(),-1);
    for(int i = 0; i < nums.size(); i++) {
        int num = nums.at(i);
        if(postions[num] != -1) {
            return num;
        } else {
            postions[num] = num;
        }
    }
    return -1;
}

bool is_valid_pair(const char a, const char b) {
    if(a == '(')
        return b == ')';
    if(a == '[')
        return b == ']';
    if(a == '{')
        return b == '}';
    return false;            
}

bool is_valid_bracket(const std::string& str) {
    if(str.empty())
        return true;    
    std::stack<char> bracket_stack;
    bracket_stack.push(str.front());
    for(size_t i = 1; i < str.size(); i++) {
        if(bracket_stack.empty()) {
            bracket_stack.push(str[i]);
            continue;
        }
        auto c = bracket_stack.top();
        if(is_valid_pair(c,str[i])) {
            bracket_stack.pop();
        } else {
            bracket_stack.push(str[i]);
        }
    }
    return bracket_stack.empty();
}

class min_stack {
public:
    min_stack() {}

    void push(int x) {
        s.push(x);
        v.push_back(x);
        order_v = v;
        std::sort(order_v.begin(),order_v.end());
    }

    void pop() {
        s.pop();
        v.pop_back();
        order_v = v;
        std::sort(order_v.begin(),order_v.end());
    }

    int top() {
        return s.top();
    }
    
    int get_min() {
        return order_v.front();
    }

private:
    std::stack<int> s;
    std::vector<int> v;
    std::vector<int> order_v;
};

//也可以维护一个每次推入元素的最小元素栈

class min_stack_2 {
public:
    min_stack_2() {}

    void push(int x) {
        s.push(x);
        if(min_s.empty())
            min_s.push(x);
        else {
            int t = min_s.top();
            min_s.push(std::min(x,t));
        }    
    }

    void pop() {
        s.pop();
        min_s.pop();
    }

    int top() {
        return s.top();
    }
    
    int get_min() {
        return min_s.top();
    }

    std::stack<int> s;
    std::stack<int> min_s;
};

std::vector<int> next_greater_num(const std::vector<int>& nums1,
                                  const std::vector<int>& nums2) {
    std::vector<int> results(nums1.size(),-1);
    for(size_t i = 0; i < nums1.size(); i++) {
        int num = nums1.at(i);
        auto index = std::find(nums2.begin(),nums2.end(),num);
        int start_index = std::distance(nums2.begin(),index);
        for(int j = start_index; j < int(nums2.size()); j++) {
            if(nums2.at(j) > num) {
                results.at(i) = nums2.at(j);
                break; 
            }
        }
    }
    return results;
}

std::string erase_outer_bracket(const std::string& str) {
    // step1: 先做原语化分解
    std::vector<std::string> primative_strings;
    int start_index = 0;
    int len = 0;
    std::stack<char> bracket_stack;
    for(size_t i = 0; i < str.size(); i++) {
        if(bracket_stack.empty() && len != 0) {
            std::string primative_string = str.substr(start_index,len);
            primative_strings.push_back(primative_string);
            start_index = i;
            len = 0;
        }
        if(bracket_stack.empty()) {
            bracket_stack.push(str[i]);
            len++;
            continue;
        }
            
        auto top = bracket_stack.top();
        if(is_valid_pair(top,str[i])) {
            bracket_stack.pop(); 
        } else {
            bracket_stack.push(str[i]);
        }
        len++;
    }
    if(bracket_stack.empty() && len != 0) {
        std::string primative_string = str.substr(start_index,len);
        primative_strings.push_back(primative_string);
    }
    // step2：对分解得到的序列字符串去掉外层括号做合并
    std::string result = "";
    for(auto& e : primative_strings) {
        result += e.substr(1,int(e.size()) - 2);
    }
    return result;
}

int baseball_score(const std::vector<std::string>& ops) {
    std::vector<int> scores;
    for(int i = 0; i < ops.size(); i++) {
        if(ops[i] == "+") {
            int last_one = scores[scores.size() - 1];
            int last_two = scores[scores.size() - 2];
            scores.push_back(last_one + last_two);
        } else if(ops[i] == "D") {
            int last_one = scores[scores.size() - 1];
            scores.push_back(last_one * 2);
        } else if(ops[i] == "C") {
            scores.pop_back();
        } else {
            scores.push_back(atoi(ops[i].c_str()));
        }
    }
    return std::accumulate(scores.begin(),scores.end(),0);
}

std::string remove_repeat(const std::string& s) {
    std::stack<char> unique_stack;
    for(size_t i = 0; i < s.size(); i++) {
        if(unique_stack.empty()) {
            unique_stack.push(s[i]);
            continue;
        }
        auto c = unique_stack.top();
        if(c == s[i])
            unique_stack.pop();
        else
        {
            unique_stack.push(s[i]);
        }
            
    }
    // 重构result
    std::string result = "";
    while(!unique_stack.empty()) {
        result = unique_stack.top() + result;
        unique_stack.pop();
    }
    return result;
}

bool backspace_compare(const std::string& s, const std::string& t) {
    std::stack<char> stack_s;
    std::stack<char> stack_t;
    for(size_t i = 0; i < s.size(); i++) {
        if(s[i] == '#') {
            if(!stack_s.empty()) {
                stack_s.pop();
            }
        } else {
            stack_s.push(s[i]);
        }
    }

    for(size_t i = 0; i < t.size(); i++) {
        if(t[i] == '#') {
            if(!stack_t.empty()) {
                stack_t.pop();
            }
        } else {
            stack_t.push(s[i]);
        }
    }

    while(!stack_s.empty()) {
        if(stack_s.top() != stack_t.top())
            return false;
        stack_s.pop();
        stack_t.pop();    
    }
    return true;
}

int remove_value(std::vector<int>& arr, const int value) {
    //找到相应的元素，原地覆盖
    int index = 0;
    for(auto& e : arr) {
        if(e != value) {
            arr[index++] = e;
        }
    }
    return index;
}

// use hash table to solve it
std::vector<int> twoSum(std::vector<int>& nums, int target) {
    //hash表储存的是数字及其对应的(位置索引+1)
    std::unordered_map<int,int> hash_table;
    for(size_t i = 0; i < nums.size(); i++) {
        int other_number = target - nums[i];
        if(hash_table[other_number] > 0) {
            return {hash_table[other_number] - 1, static_cast<int>(i)};
        } else {
            // 把当前数先加入哈希表
            hash_table[nums[i]] = i + 1;
        }
    }
    std::cout<<"no pair found"<<std::endl;
    return {};
}

int maxEqualRowsAfterFlips(std::vector<std::vector<int>>& matrix) {
    //翻转后的最多行 = 某一行出现的个数 + 其补行出现的个数
    std::map<std::string,int> has; //遍历使用map来进行遍历
    for(size_t i = 0; i < matrix.size(); i++) {
        std::string tmp = "";
        for(size_t j = 0; j < matrix[i].size(); j++) {
            tmp += std::to_string(matrix[i][j]);
        }
        has[tmp] ++;
    }
    int result = 0;
    for(auto iter = has.begin(); iter != has.end(); iter++) {
        std::string tmp = iter->first;
        std::string rev = "";
        for(size_t i = 0; i < tmp.size();i++) {
            if(tmp[i] == '1')
                rev += '0';
            else
                rev += '1';    
        }
        //原本的每一行都对应一个最多行的个数
        result = std::max(result, iter->second + has[rev]);
    }
    return result;        
}

int singleNumber(std::vector<int>& nums) {
    std::map<int,int> has;
    for(size_t i = 0; i < nums.size(); i++) {
        has[nums[i]]++;
    }
    for(auto iter = has.begin(); iter != has.end(); iter++) {
        if(iter->second == 1)
            return iter->first;
    }
    return nums.front();
}

std::vector<int> intersection(std::vector<int>& nums1, std::vector<int>& nums2) {
    //使用哈希来判断数组交集,准备一半(O(N))
    std::unordered_map<int,int> has1;
    for(auto& num : nums1) {
        has1[num]++;
    }
    std::set<int> results_set;
    for(auto& num : nums2) {
        if(has1[num] > 0) {
            results_set.insert(num);
        }
    }
    std::vector<int> results;
    for(auto& num : results_set) {
        results.push_back(num);
    }
    return results;
}

int fourSumCount(std::vector<int>& A, std::vector<int>& B, 
                 std::vector<int>& C, std::vector<int>& D) {
    //准备一半 ，哈希AB和CD的和
    std::map<int,std::vector<std::pair<int,int>>> hash_AB;
    std::map<int,std::vector<std::pair<int,int>>> hash_CD;
    for(size_t i = 0; i < A.size(); i++) {
        for(size_t j = 0; j < B.size(); j++) {
            int sum = A[i] + B[j];
            hash_AB[sum].push_back({i,j});
        }
    }
    for(size_t i = 0; i < C.size(); i++) {
        for(size_t j = 0; j < D.size(); j++) {
            int sum = C[i] + D[j];
            hash_CD[sum].push_back({i,j});
        }
    }
    int result = 0;
    for(auto iter = hash_AB.begin(); iter != hash_AB.end(); iter++) {
        int value = -(iter->first);
        if(hash_CD.count(value) != 0)
            result += iter->second.size() * hash_CD[value].size();
    }
    return result;    
}

class TinyUrl {
public:
    std::string encode(std::string long_url) {
        return long_url;
    }

    std::string decode(std::string short_url) {
        return short_url;
    }
};


int subarraySum(std::vector<int>& nums, int k) {
    // 计算累积分布
    std::map<int,int> accumulate_map;
    accumulate_map[0] = nums.front();
    for(size_t i = 1; i < nums.size(); i++) {
        accumulate_map[i] = nums[i] + accumulate_map[i-1];
    }
    std::map<int,std::vector<int>> accumulate_value;
    for(auto iter = accumulate_map.begin(); iter != accumulate_map.end(); iter++) {
        int index = iter->first;
        int sum = iter->second;
        accumulate_value[sum].push_back(iter->first);
    }
    int result = 0;
    if(accumulate_value.count(k) != 0)
        result += accumulate_value[k].size();
    int start_index = 1;
    for( ; start_index < int(nums.size()); start_index++) {
        int accumulate_sum = k + accumulate_map[start_index - 1];
        if(accumulate_value.count(accumulate_sum) != 0) {
            auto v = accumulate_value[accumulate_sum];
            result += std::count_if(v.begin(),v.end(),[&](int i) {
                return i >= start_index;
            });
        }
            
    }
    return result;
}


// todo use hash
int subarraysDivByK(std::vector<int>& A, int K) {
    std::map<int,int> accumulate_map;
    accumulate_map[0] = A.front();
    for(size_t i = 1; i < A.size(); i++) {
        accumulate_map[i] = A[i] + accumulate_map[i-1];
    }
    std::map<int,std::vector<int>> accumulate_value;
    for(auto iter = accumulate_map.begin(); iter != accumulate_map.end(); iter++) {
        int index = iter->first;
        int sum = iter->second;
        accumulate_value[sum % K].push_back(iter->first);
    }
    int result = 0;
    if(accumulate_value.count(0) != 0)
        result += accumulate_value[0].size();  
    int start_index = 1;
    for(; start_index < A.size(); start_index++) {
        int prenum_sum_mod = accumulate_map[start_index - 1];
        for(int j = start_index; j < A.size();j ++) {
            int value = accumulate_map[j] - prenum_sum_mod;
            if(value % K == 0)
                result++;
        }
    }
    return result;
}

struct TreeNode {
    int val;
    TreeNode* left_child;
    TreeNode* right_child;
    TreeNode(int v) : val(v),left_child(nullptr),right_child(nullptr) {}
    TreeNode(int v, TreeNode* left, TreeNode* right) :
             val(v),left_child(left),right_child(right) {};
};

//递归结构恢复二叉树，哈希表进行元素查询
class FindElements {
public:
    FindElements(TreeNode* root) {
        recover(root,0);
    }
    
    bool find(int target) {
        return has[target] > 0;
    }

    void recover(TreeNode* root,int val) {
        if(root == nullptr)
            return;
        root->val = val;
        has[val]++;
        recover(root->left_child,2*val + 1);
        recover(root->right_child,2*val + 2);
    }

    // default value = 0;
    std::map<int,int> has;
};

void moveZeroes(std::vector<int>& nums) {
    int pos = 0;
    for(auto& num : nums) {
        if(num != 0) {
            nums[pos] = num;
            pos++;
        }
    }
    for(int i = pos; i < nums.size(); i++) {
        nums[i] = 0;
    }    
}

int removeDuplicates(std::vector<int>& nums) {
    auto iter = std::unique(nums.begin(),nums.end());
    return std::distance(nums.begin(),iter);
}

int removeDuplicates_2(std::vector<int>& nums) {
    if(nums.empty())
        return 0;
    int pos = 1;
    int repeat_times = 1;
    for(size_t i = 1; i < nums.size(); i++) {
        if(nums[i] != nums[i-1]) {
            nums[pos++] = nums[i];
            repeat_times = 1;
        } else {
            if(repeat_times == 1) {
                nums[pos++] = nums[i];
                repeat_times = 2;
            } else {
                continue;
            }
        }
    }
    return pos;
}

void reverseString(std::vector<char>& s) {
    std::string a;
    std::reverse(s.begin(),s.end());
}

//翻转索引 i 和 j 之间的数据，包括索引 i 和 j
void reverseStr(std::string& s, int i, int j) {
    int ii = i;
    int jj = j;
    while(ii < jj) {
        std::swap(s[ii],s[jj]);
        ii++;
        jj--;
    }
}

//索引的递增
std::string reverseStr(std::string s, int k) {
    std::string result = s;
    if(k >= result.size())
        reverseStr(result,0,result.size()-1);
    else
        reverseStr(result,0,k-1);
    int start_index = 2 * k;
    int end_index = start_index + 2 * k - 1;
    while(end_index < result.size()) {
        reverseStr(result,start_index, start_index + k - 1);
        start_index += 2 * k;
        end_index += 2 * k;
    }
    if(start_index + k  >= result.size())
        reverseStr(result,start_index,result.size() - 1);
    else
        reverseStr(result,start_index,start_index + k - 1);
    return result;
}

//无符号整数中1的个数
int hammingWeight(uint32_t n) {
    int cnt = 0;
    while(n > 0) {
        n = n & (n-1);
        cnt++;
    }
    return cnt;   
}

bool isPowerOfTwo(int n) {
    int cnt = 0;
    while(n > 0) {
        cnt += n & 1;
        n = n >> 1;
    }
    return cnt == 1;        
}

int hammingDistance(int x, int y) {
  int diff = x ^ y;
  int result = 0;
  while(diff) {
      result += diff & 1;
      diff >>= 1;
  }
  return result;
}

std::vector<std::vector<int>> threeSum(std::vector<int>& nums) {
    std::vector<std::vector<int>> results;
    if(nums.size() < 3)
        return results;
    std::map<int,std::vector<std::pair<int,int>>> two_sum_map;
    for(size_t i = 0; i < nums.size(); i++) {
        for(size_t j = i + 1; j < nums.size(); j++) {
            int sum = nums[i] + nums[j];
            two_sum_map[sum].push_back({i,j});
        }
    }
    std::set<std::vector<int>> tmp_results;
    for(size_t i = 0; i < nums.size() - 2; i++) {
        if(two_sum_map.count(-nums[i]) != 0) {
            auto& pairs = two_sum_map[-nums[i]];
            for(auto& pair : pairs) {
                if(pair.first <= i || pair.second <= i)
                    continue;
                std::vector<int> tmp{nums[i],nums[pair.first],nums[pair.second]};
                std::sort(tmp.begin(),tmp.end());
                tmp_results.insert(tmp);   
            }
        }
    }
    for(auto& result : tmp_results) {
        results.push_back(result);
    }  
    return results;
}

std::vector<int> dailyTemperatures(std::vector<int>& T) {
    // std::vector<int> results(T.size(), 0);
    // for(size_t i = 0; i < T.size()-1; i++) {
    //     for(int j = i + 1; j < T.size(); j++) {
    //         if(T[j] > T[i]) {
    //             results[i] = j - i;
    //             break;
    //         }
    //     }
    // }
    // return results;
    
    // 维护一个单调递减栈，有大的数就可以更新结果
    // 递减栈
    std::vector<int> results(T.size(), 0);
    std::stack<std::pair<int,int>> temper_stack; // index and temperatures
    temper_stack.push({0, T.front()});
    for(size_t i = 1; i < T.size(); i++) {
        if(T[i] <= temper_stack.top().second) {
            temper_stack.push({i,T[i]});
            continue;
        }
        while(!temper_stack.empty() && T[i] > temper_stack.top().second) {
            results[temper_stack.top().first] = i - temper_stack.top().first;
            temper_stack.pop();
        }
        temper_stack.push({i,T[i]});
    }
    return results;
}

/*
给定状态下出现观测的概率
*/

std::vector<int> plusOne(std::vector<int>& digits) {
    std::vector<int> results;
    int over = 0;
    int num = 0;
    for(int i = (digits.size() - 1); i >=0 ; i--) {
        if(i == digits.size() - 1)
            num = digits.at(i) + 1 + over;
        else
        {
            num = digits.at(i) + over;
        }
        over = num / 10;
        results.push_back(num % 10);
    }
    if(over != 0)
        results.push_back(over);
    std::reverse(results.begin(),results.end());
    return results;    
}

int validIndex(const int i, const int j, const std::vector<std::vector<int>>& M) {
    //索引有效，返回像素值，否则为-1
    if(i >= 0 && i < M.size() && j >= 0 && j < M.front().size())
         return M[i][j];
    return -1;     
}

std::vector<std::vector<int>> imageSmoother(std::vector<std::vector<int>>& M) {
    int rows = M.size();
    int cols = M.front().size();
    std::vector<std::vector<int>> smooth_image(rows,std::vector<int>(cols,0));
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            int valid_count = 0;
            int sum = 0;
            for(int k = -1; k <=1; k++) {
                for(int l = -1; l <= 1; l++) {
                    if(validIndex(i+k,j+l,M) != -1) {
                        sum += validIndex(i+k,j+l,M);
                        valid_count++;
                    }
                }
            }
            std::cout<< sum <<std::endl;
            smooth_image[i][j] = std::floor(sum / valid_count);
        }
    }
    return smooth_image;
}

int majorityElement(std::vector<int>& nums) {
    int maj_num = 0;
    int maj_count = 0;
    std::unordered_map<int,int> has;
    for(auto& num : nums) {
        has[num] ++;
        if(has[num] > maj_count) {
            maj_count = has[num];
            maj_num = num;
        }
    }
    return maj_num;
}

struct ListNode {
    int val;
    ListNode* next;
    ListNode(int v) : val(v),next(nullptr) {}
};

// 分割链表，使得小于x的节点在大于等于x节点的左边
ListNode* partition(ListNode* head, int x) {
    ListNode* p = head;
    ListNode* dummysmall = nullptr;
    ListNode* dummylarge = nullptr;
    ListNode* dummyp = nullptr;
    ListNode* dummyq = nullptr;
    while(p != nullptr) {
        if(p->val < x) {
            if(dummysmall == nullptr) {
                dummysmall = p;
                dummyp = dummysmall;
            } else {
                dummyp->next = p;
                dummyp = p;
            }
        } else {
            if(dummylarge == nullptr) {
                dummylarge = p;
                dummyq = dummylarge;
            } else {
                dummyq->next = p;
                dummyq = p;
            }
        }
        //传递好之后，断开之前的节点
        p = p->next;
        if(dummyp)
            dummyp->next = nullptr;
        if(dummyq)
            dummyq->next = nullptr;    
    }
    if(dummysmall == nullptr)
        return dummylarge;
    dummyp->next = dummylarge;
    return dummysmall;    
}

//3000ms内的请求次数统计
class RecentCounter {
public:
    RecentCounter() {
        
    }
    
    int ping(int t) {
        int count = 1;
        for(int i = int(timestamps.size()-1); i >= 0; i--) {
            std::cout<<"i: "<<i<<std::endl;
            if(timestamps[i] >= (t - 3000))
                count++;
            else
                break;    
        }
        timestamps.push_back(t);
        return count;
    }

    std::vector<int> timestamps;
};

//基于线性表的循环队列实现
class MyCircularQueue {
public:
    /** Initialize your data structure here. Set the size of the queue to be k. */
    MyCircularQueue(int k) {
        data_nums = 0;
        size = k;
        front_index = 0;
        end_index = 0;
        queue_data = std::vector<int>(size,-1);
    }
    
    /** Insert an element into the circular queue. Return true if the operation is successful. */
    bool enQueue(int value) {
        if(isFull())
            return false;
        queue_data[end_index] = value;
        end_index = (end_index + 1) % size;
        data_nums++;
        return true;
    }
    
    /** Delete an element from the circular queue. Return true if the operation is successful. */
    bool deQueue() {
        if(isEmpty())
            return false;
        front_index = (front_index + 1) % size;
        data_nums--;
        return true;
    }
    
    /** Get the front item from the queue. */
    int Front() {
        if(isEmpty())
            return -1;
        return queue_data[front_index];
    }
    
    /** Get the last item from the queue. */
    int Rear() {
        if(isEmpty())
            return -1;
        int rear_index = end_index - 1;
        if(rear_index < 0)
            rear_index += size;
        return queue_data[rear_index];
    }
    
    /** Checks whether the circular queue is empty or not. */
    bool isEmpty() {
        return data_nums == 0;
    }
    
    /** Checks whether the circular queue is full or not. */
    bool isFull() {
        return data_nums == size;
    }

    std::vector<int> queue_data;
    int data_nums;
    int front_index;
    int end_index;
    int size;
};

// 循环双端队列
class MyCircularDeque {
public:
    /** Initialize your data structure here. Set the size of the deque to be k. */
    MyCircularDeque(int k) {
        queue_data = std::vector<int>(k,-1);
        data_nums = 0;
        front_index = -1;
        end_index = 0;
        size = k;
    }
    
    /** Adds an item at the front of Deque. Return true if the operation is successful. */
    bool insertFront(int value) {
        if(isFull())
            return false;
        if(front_index == -1) {
            if(isEmpty()) {
                queue_data[0] = value;
                front_index = 0; // front_index 指向当前的队首元素
                end_index = 1;
            } else {
                queue_data.back() = value;
                front_index = size - 1;
            }
        } else {
            front_index -= 1;
            if(front_index < 0)
                front_index += size;
            queue_data[front_index] = value;    
        }    
        data_nums++;
        return true;
    }
    
    /** Adds an item at the rear of Deque. Return true if the operation is successful. */
    bool insertLast(int value) {
        if(isFull())
            return false;
        queue_data[end_index] = value;
        end_index = (end_index + 1) % size;
        data_nums++;
        return true;
    }
    
    /** Deletes an item from the front of Deque. Return true if the operation is successful. */
    bool deleteFront() {
        if(isEmpty())
            return false;
        front_index = (front_index + 1) % size;
        data_nums--;
        return true; 
    }
    
    /** Deletes an item from the rear of Deque. Return true if the operation is successful. */
    bool deleteLast() {
        if(isEmpty())
            return false;
        end_index--;
        if(end_index < 0)
            end_index += size;
        data_nums--;
        return true;     
    }
    
    /** Get the front item from the deque. */
    int getFront() {
        if(isEmpty())
            return -1;
        return queue_data[front_index];    
    }
    
    /** Get the last item from the deque. */
    int getRear() {
        if(isEmpty())
            return -1;
        int index = end_index - 1;
        if(index < 0)
            index += size;
        return queue_data[index];        
    }
    
    /** Checks whether the circular deque is empty or not. */
    bool isEmpty() {
        return data_nums == 0;
    }
    
    /** Checks whether the circular deque is full or not. */
    bool isFull() {
        return data_nums == size;
    }

    std::vector<int> queue_data;
    int data_nums;
    int front_index;
    int end_index;
    int size;
};

// binary tree
// left == left_child right = right_child
std::vector<int> inorderTraversal(TreeNode* root) {
    if(!root)
        return {};
    std::vector<int> result = inorderTraversal(root->left_child);
    result.push_back(root->val);
    auto right_result = inorderTraversal(root->right_child);
    result.insert(result.end(),right_result.begin(),right_result.end());
    return result; 
}

bool isSameTree(TreeNode* p, TreeNode* q) {
    if(p == nullptr && q == nullptr)
        return true;
    if(p == nullptr & q != nullptr)
        return false;
    if(p != nullptr && q == nullptr)
        return false;
    if(p ->val != q->val)
        return false;
    return isSameTree(p->left_child,q->left_child) && 
           isSameTree(p->right_child,q->right_child);                      
}

//递归的判断是否为相同的树状结构
bool isSubtree(TreeNode* s, TreeNode* t) {
  if(s == nullptr)
    return false;
  if(isSameTree(s, t))
    return true;
  return isSubtree(s->left_child, t) || isSubtree(s->right_child, t);
}

//左树和右树是否形成对称结构
bool isSymmetric(TreeNode* left_tree, TreeNode* right_tree) {
    if(left_tree == nullptr && right_tree == nullptr)
        return true;
    if(left_tree == nullptr & right_tree != nullptr)
        return false;
    if(left_tree != nullptr && right_tree == nullptr)
        return false;
    if(left_tree->val != right_tree->val)
        return false;
    return isSymmetric(left_tree->left_child,right_tree->right_child) &&
           isSymmetric(left_tree->right_child,right_tree->left_child);        
}

bool isSymmetric(TreeNode* root) {
    if(root == nullptr)
        return true;
    return isSymmetric(root->left_child,root->right_child);    
}

//二叉树的最大深度
int maxDepth(TreeNode* root) {
    if(root == nullptr)
        return 0;
    return 1 + std::max(maxDepth(root->left_child), maxDepth(root->right_child));        
}

// 到叶子节点的最小深度
int minDepth(TreeNode* root) {
    if(root == nullptr)
        return 0;
    if(root->left_child == nullptr)
        return 1 + minDepth(root->right_child);
    if(root->right_child == nullptr)
        return 1 + minDepth(root->left_child);
    //左右都有才进行深度大小的比较            
return 1 + std::min(minDepth(root->left_child),minDepth(root->right_child));            
}

// 左右子树高度上是否平衡
// 先求每个节点上的深度
bool isBalanced(TreeNode* root) {
    if(root == nullptr)
        return true;
    int left_depth = maxDepth(root->left_child);
    int right_depth = maxDepth(root->right_child);
    if(std::abs(left_depth - right_depth) > 1)
        return false;
    return isBalanced(root->left_child) && isBalanced(root->right_child);    
}

// 最大平均数的连续子数组
double findMaxAverage(std::vector<int>& nums, int k) {
    // brute force sulation
    // int max_sum = std::numeric_limits<int>::min();
    // for(int i = 0; i <= nums.size() - k; i++) {
    //     int sum = 0;
    //     for(int j = i; j < i + k; j++) {
    //         sum += nums[j];
    //     }
    //     if(sum > max_sum) {
    //         max_sum = sum;
    //     }
    // }
    // return max_sum / double(k);

    // sliding window sulation
    int start_sum = 0;
    for(int i = 0; i < k; i++) {
        start_sum += nums[i];
    }
    int max_sum = start_sum;
    int start_window_index = 1;
    for(; start_window_index <= nums.size() - k; start_window_index++) {
        int tmp_sum = start_sum - nums[start_window_index - 1] + nums[start_window_index + k - 1];
        if(tmp_sum > max_sum)
            max_sum = tmp_sum;
        start_sum = tmp_sum; // 上一阶段的子数组之和    
    }
    return max_sum / double(k);
}

int trap(std::vector<int>& height) {
    std::vector<int> left_high(height.size(), 0);
    std::vector<int> right_high(height.size(),0);
    int left_max = 0;
    for(int i = 0; i < int(height.size()); i++) {
        left_high.at(i) = left_max;
        left_max = std::max(left_max,height[i]);
    }
    int right_max = 0;
    for(int  i = height.size() - 1; i >= 0; i--) {
        right_high[i] = right_max;
        right_max = std::max(right_max, height[i]);
    }

    int result = 0;
    for(int i = 1; i < height.size() - 1; i++) {
        int current_height = height[i];
        int left_max_height = left_high[i];
        int right_max_height = right_high[i];
        result += std::max(0, std::min(left_max_height,right_max_height) - current_height);
    }
    return result;
}

int findPeakElement(std::vector<int>& nums) {
    for(int i = 0; i < nums.size(); i++) {
        if(i == 0) {
            if(nums[i] > nums[i+1])
                return 0;
        }
        if(i == nums.size() - 1) {
            if(nums[i] > nums[i-1])
                return i-1;
        }
        if(nums[i] > nums[i-1] && nums[i] > nums[i+1])
            return i;
        continue;
    }
    return 0;
}

void zhushi() {
    #if 0
    std::cout<<"zhushi"<<std::endl;
    #endif

    #if 1
    std::cout<<"not zhushi"<<std::endl;
    #endif
}

// 联合体会覆盖内存空间值
union a_union {
    struct B {
        int x;
        int y;
    } b;

    int k;
};

enum weekday {
    sun, mon, the, wed, thu,may,sat
};

struct student
{
    int num;
    char name[20];
    char gender;
};

//使用类的前向声明定义时，只允许使用指针这样不完整的定义
class B;

class A {
public:

private:
B* b;
};

class Application
{ 
public:
    static void f(); 
    static void g();
private:
    static int global;
};

int Application::global=0;

void Application::f() {  global=5; }
void Application::g() {  std::cout<<global<<std::endl;}

// 常对象只能调用其对应的常成员函数
// void print() const; const A a;

// 根据实参调用来确定模版函数中参数的类型，然后编译器生成相应类型的模版函数
template<typename T>
void sort(T& a, int n){
    for(int i = 0; i < n; i++) {
        for(int j = i+1; j < n; j++) {
            if(a[j] > a[i]) {
                int tmp = a[i];
                a[i] = a[j];
                a[j] = tmp;
            }
        }
    }
}

template<typename T>
void display(T& a,int n){
    for(int i = 0; i < n;i++) {
        std::cout<<a[i]<<" ";
    }
    std::cout<<std::endl;
}

template<typename T, int MAXSIZE>
class Stack {
    private:
    int top = -1;
    T elem[MAXSIZE];

    public:
    bool empty() {
        if(top <= -1)
            return 1;
        return 0;    
    }

    bool full() {
        if(top >= (MAXSIZE-1))
            return 1;
        return 0;    
    }

    void push(T e);
    T pop();
};

// 模版类定义
template<typename T, int MAXSIZE>
void Stack<T, MAXSIZE>::push(T e) {
    if(full()) {
        std::cout<<"already full"<<std::endl;
        return;
    }
    elem[++top] = e;
}

template<typename T, int MAXSIZE>
T Stack<T, MAXSIZE>::pop() {
    if(empty()) {
        std::cout<<"already empty"<<std::endl;
        return;
    }
    return elem[top--];
}

template<typename T>
using Vec = std::vector<T, std::allocator<T>>;

//没有explicit声明，类对象会进行隐式转换
//加入explicit声明，类对象不会进行隐式转换 

//final 放在类后面表示该类不能被继承
//final 放在函数后面表示该函数不能被override

//virtual override会做重载虚函数检查

void findOdd(unsigned long long start, unsigned long long end) {
    unsigned long long odd_count = 0;
    for(unsigned long long i = start; i < end; i++) {
        if((i & 1) == 1)
            odd_count++;
    }
}

void findEven(unsigned long long start, unsigned long long end) {
    unsigned long long even_count = 0;
    for(unsigned long long i = start; i < end; i++) {
        if((i & 1) == 0)
            even_count++;
    }
}

void compare_time() {
    unsigned long long start = 0;
    unsigned long long end = 19000000;
    auto start_time = std::chrono::high_resolution_clock::now();
    findOdd(start, end);
    findEven(start, end);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = 
        std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout<<"un thread time: "<< duration.count() / 1000000.0 <<"s";

    start_time = std::chrono::high_resolution_clock::now();
    std::thread t1(findOdd, start, end);
    std::thread t2(findEven, start, end);
    end_time = std::chrono::high_resolution_clock::now();
    duration = 
        std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout<<"thread time: "<< duration.count() / 1000000.0 <<"s";
}

class Base {
    public:
    static void func(int n) {
        while(n--) {
            std::cout<< n << " ";
        }
    }
};

void run(int count) {
    while (count-- > 0) {
        std::cout << count << std::endl;
    }
    std::this_thread::sleep_for(std::chrono::seconds(3));
}

//数据竞争导致数据读写的错误
int sum = 0;
std::mutex m;
void countgold() {
    int i; //local to each thread
    for (i = 0; i < 10000000; i++) {
        sum += 1;
    }
}

int string2int(std::string str) {
    return std::stoi(str);
}

ListNode* reverseList(ListNode* head) {
    if(head == nullptr || head->next == nullptr)
        return head;
    ListNode* pre_node = head;
    ListNode* cur_node = pre_node->next;
    pre_node->next = nullptr;

    while(cur_node->next != nullptr) {
        ListNode* next_node = cur_node->next;
        cur_node->next = pre_node;
        pre_node = cur_node;
        cur_node = next_node;
    }
    cur_node->next = pre_node;
    return cur_node;    
}

ListNode* swapPairs(ListNode* head) {
    if(head == nullptr || head->next == nullptr)
        return head;
    ListNode* pre = head;
    ListNode* cur = pre->next;
    ListNode* result = head->next;
    ListNode* next = cur->next;

    while(pre != nullptr && cur != nullptr && next != nullptr) {
        cur->next = pre;
        if(next->next != nullptr)
            pre->next = next->next;
        else
        {
            pre->next = next;
        }
            

        pre = next;
        if(pre == nullptr)
            break;
        cur = pre->next;
        if(cur == nullptr)
            break;
        next = cur->next;
    }

    if(cur == nullptr)
        return result;

    if(next == nullptr) {
        cur->next = pre; 
        pre->next = nullptr;
    }
           

   return result;
        
}

bool hasCycle(ListNode *head) {
    if(head == nullptr || head->next == nullptr)
        return false;
    ListNode* Fast = head->next;
    ListNode* Slow = head;
    while(Fast != nullptr && Slow != nullptr) {
        if(Fast == Slow)
            return true;
        Slow = Slow->next;
        if(Fast->next == nullptr)
            return false;
        Fast = Fast->next->next;        
    }

    return false;
}

class KthLargest {
public:
    KthLargest(int k, std::vector<int>& nums) {
        k_ = k;
        for(int i : nums) {
            if(q.size() < k)
                q.push(i);
            else
            {
                if(i > q.top()) {
                    q.pop();
                    q.push(i);
                }
                    
            }
            
        }
    }
    
    int add(int val) {
        if(q.size() < k_) {
            q.push(val);
            return q.top();
        }
        if(val <= q.top())
            return q.top();
        q.pop();    
        q.push(val);
        return q.top();    
    }
    
    std::priority_queue<int, std::vector<int>, std::greater<int> > q;
    int k_;
};

/**
 * Your KthLargest object will be instantiated and called as such:
 * KthLargest* obj = new KthLargest(k, nums);
 * int param_1 = obj->add(val);
 */

//用双端队列deque求解sliding window问题
std::vector<int> maxSlidingWindow(std::vector<int>& nums, int k) {
    std::vector<int> results;
    return results;
}

bool isAnagram(std::string s, std::string t) {
    std::unordered_map<char, int> char_count;
    for(char c : s) {
        if(char_count.count(c) == 0)
            char_count[c] = 1;
        else
        {
            char_count[c] ++;
        }      
    }
    for(char c : t) {
        if(char_count[c] == 0)
            return false;
        else
        {
            char_count[c]--;
        }    
    }
    for(auto e : char_count) {
        if(e.second != 0)
            return false;
    }

    return true;
}

std::vector<std::vector<int>> threeSum_1(std::vector<int>& nums) {
    std::vector<std::vector<int>> results;

    std::map<int,int> num_count;
    for(int& num : nums) {
        if(num_count.count(num) == 0)
            num_count[num] = 1;
        else {
            num_count[num] += 1;
        }     
    }

    for(int i = 0; i < nums.size(); i++) {
        for(int j = i+1; j < nums.size(); j++) {
            int target_num = -(nums[i] + nums[j]);
            int count = 1;
            if(target_num == nums[i])
                count++;
            if(target_num == nums[j])
                count++;
            if(num_count[target_num] == count)
                results.push_back({nums[i], nums[j], target_num});        
        }
    }
    return results;
}

//中序遍历的二叉搜索树是一个升序的结构
void inorder_tranverce(TreeNode* root, std::vector<int>& results) {
    if(root == nullptr)
        return;
    inorder_tranverce(root->left_child, results);
    results.push_back(root->val);
    inorder_tranverce(root->right_child, results);            
}

//找二叉搜索树中的众数
std::vector<int> findMode(TreeNode* root) {
  std::vector<int> results;
  inorder_tranverce(root, results);
  if(results.size() < 2)
    return results;
  // value and count
  std::vector<std::pair<int, int>> final_results;
  final_results.push_back({results.front(), 1});
  for(int i = 1; i < results.size(); i++) {
    if(results[i] == results[i-1])
      final_results.back().second++;
    else {
        final_results.push_back({results[i], 1});
    }  
  }

  std::sort(final_results.begin(), final_results.end(), [](std::pair<int, int> a, 
  std::pair<int, int> b){
    return a.second > b.second;
  });
  
  std::vector<int> mode_result;
  mode_result.push_back(final_results[0].first);
  for(int i = 1; i < final_results.size(); i++) {
    if(final_results[i].second != final_results[i-1].second)
      break;
    else
      mode_result.push_back(final_results[i].first);  
  }
  return mode_result;
}


int getMinimumDifference(TreeNode* root) {
  std::vector<int> results;
  inorder_tranverce(root, results);
  int result = std::numeric_limits<int>::max();
  for(int i = 0; i < results.size()-1; i++) {
      result = std::min(result, std::abs(results[i+1] - results[i]));
  }
  return result;
}

bool isValidBST(TreeNode* root) {
    std::vector<int> inorder_results;
    inorder_tranverce(root, inorder_results);
    for(size_t i = 0; i < inorder_results.size() - 1; i++) {
        if(inorder_results[i+1] <= inorder_results[i])
            return false;
    }
    return true;
}

//二叉搜索树的最近公共祖先
TreeNode* lowestCommonAncestor_BST(TreeNode* root, TreeNode* p, TreeNode* q) {
    if(root == nullptr || p == nullptr || q == nullptr)
        return root;
    if(p->val < root->val && q->val > root->val)
        return root;
    if(p->val < root->val && q->val < root->val)
        return lowestCommonAncestor_BST(root->left_child, p, q);
    if(p->val > root->val && q->val > root->val)
        return lowestCommonAncestor_BST(root->right_child, p, q);
    return root;
        
}

//二叉树的最近公共祖先
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    if(root == nullptr || root == p || root == q)
        return root;
    TreeNode* left = lowestCommonAncestor(root->left_child, p, q);
    TreeNode* right = lowestCommonAncestor(root->right_child, p, q);
    if(left == nullptr)
        return right;
    if(right == nullptr)
        return left;
    return root;            
}

//递归计算
double myPow(double x, int n) {
    //return std::pow(x,n);
    if(n == 0 && x != 0)
        return 1;
    if(n < 0)
        return 1 / myPow(x, -n);
    double pow = 1;
    if(n % 2 == 1)
        return x * myPow(x * x, n / 2);
    return myPow(x * x, n / 2);      
}

//二叉树层次遍历输出
//把这一层的先消耗掉，再推入下一层的元素
std::vector<std::vector<int>> levelOrder(TreeNode* root) {
    std::vector<std::vector<int>> results;
    if(root == nullptr)
        return results;
    std::queue<TreeNode*> q;
    q.push(root);
    while(!q.empty()) {
        std::vector<int> result;
        std::vector<TreeNode*> current_level_nodes;
        //记录当前层级的节点
        while(!q.empty()) {
            result.push_back(q.front()->val);
            current_level_nodes.push_back(q.front());
            q.pop();
        }
        results.push_back(result);
        for(auto& node : current_level_nodes) {
            if(node->left_child != nullptr)
                q.push(node->left_child);
            if(node->right_child != nullptr)
                q.push(node->right_child);    
        }
    }
    return results;    
}

//left right代表已经用了的括号的个数
void gen_helper(int n, int left, int right, std::string str ,std::vector<std::string>& results) {
    if(left == n && right == n) {
        results.push_back(str);
        return;
    }

    if(left < n) {
        gen_helper(n, left+1, right, str + "(", results);
    }
    if(right < n && right < left) {
        gen_helper(n, left, right+1,str + ")",results);
    }
}

//生成有效的括号集合
std::vector<std::string> generateParenthesis(int n) {
    std::vector<std::string> results;
    std::string str = "";
    gen_helper(n, 0, 0, str, results);
    return results;
}

int sumNums(int n) {
    bool b = (n > 0) && (n += sumNums(n-1));
    return n;
}

bool isPowerOfTwo2(int n) {
    if(n <= 0)
    return false;
    return (n & (n-1)) == 0;
}

int hammingWeight2(uint32_t n) {
    int count = 0;
    while(n) {
        count++;
        n = n & (n-1);
    }
    return count;
}

int singleNumber2(std::vector<int>& nums) {
    std::unordered_map<int, int> num_count;
    for(int i = 0; i < nums.size(); i++) {
        if(num_count.count(nums[i]) == 0) {
            num_count[nums[i]] = 1;
        } else {
            num_count[nums[i]]++;
        }
    }
    for(auto iter = num_count.begin(); iter != num_count.end(); iter++){
        if(iter->second == 1)
          return iter->first;
    }
    return -1;
}

int singleNumber3(std::vector<int>& nums) {
    //去重 * 3 ,相减再除以2
    return 0;
}

//让偶数排到奇数之前
//记录奇数的索引，遇到偶数的时候不断进行交换
std::vector<int> sortArrayByParity(std::vector<int>& A) {
  std::vector<int> odd_index;
  for(int i = 0; i < A.size(); i++) {
      if(((A[i] % 2) == 0) && !odd_index.empty()) {
          std::swap(A[i], A[odd_index.front()]);
          odd_index.erase(odd_index.begin());
      }
      if((A[i] % 2) == 1)
        odd_index.push_back(i);
  }
  return A;
}

// o(n^2) soluation
int maxSubArray1(std::vector<int>& nums) {
    int len = nums.size();
    int dp[len][len];
    dp[0][0] = nums[0];
    int max_result = dp[0][0];
    for(int j = 1; j < nums.size(); j++) {
        dp[0][j] = dp[0][j-1] + nums[j];
        max_result = std::max(max_result, dp[0][j]);
    }
    for(int i = 1; i < nums.size(); i++) {
        dp[i][i] = nums[i];
        max_result = std::max(max_result, dp[i][i]);
    }
    for(int i = 1; i < nums.size(); i++) {
        for(int j = i+1; j < nums.size(); j++) {
            dp[i][j] = dp[i][j-1] + nums[j];
            max_result = std::max(max_result, dp[i][j]);
        }
    }
    return max_result;
}

// dp[i]=max(nums[i], dp[i−1]+nums[i])
// dp[i]表示以i结尾的序列
int maxSubArray(std::vector<int>& nums) {
  int dp[nums.size()];
  dp[0] = nums[0];
  int max_result = dp[0];
  for(int i = 1; i < nums.size(); i++) {
      dp[i] = std::max(dp[i-1] + nums[i], nums[i]);
      max_result = std::max(max_result, dp[i]);
  }
  return max_result;
}

//dp[i] = max(dp[j]+1，dp[k]+1，dp[p]+1，.....)
//dp[i]表示以i结尾的最长上升序列的长度, 最小长度为1
int lengthOfLIS(std::vector<int>& nums) {
  int dp[nums.size()];
  dp[0] = 1;
  int max_result = dp[0];
  for(int i = 1; i < nums.size(); i++) {
      dp[i] = 1;
      for(int j = 0; j < i; j++) {
          if(nums[j] < nums[i]) {
              dp[i] = std::max(dp[i], dp[j] + 1);
          }
      }
      max_result = std::max(max_result, dp[i]);
  }
  return max_result;
}

//三角形的最小路径和
int minimumTotal(std::vector<std::vector<int>>& triangle) {
    int row = triangle.size();
    int dp[row][row];
    dp[0][0] = triangle[0][0];
    dp[1][0] = dp[0][0] + triangle[1][0];
    dp[1][1] = dp[0][0] + triangle[1][1];
    for(int i = 2; i < row; i++) {
        for(int j = 0; j < row; j++) {
            if(j == 0) {
                dp[i][j] = dp[i-1][j] + triangle[i][j];
            } else if(j == i) {
                dp[i][j] = dp[i-1][j-1] + triangle[i][j];
            } else {
                dp[i][j] = std::min(dp[i-1][j-1] + triangle[i][j],
                                    dp[i-1][j] + triangle[i][j]);
            }
        }
    }
    int min_result = dp[row-1][0];
    for(int i = 1; i < row; i++) { 
        min_result = std::min(min_result, dp[row-1][i]);
    }
    return min_result;
}

int minPathSum(std::vector<std::vector<int>>& grid) {
    int row = grid.size();
    int dp[row][row];
    dp[0][0] = grid[0][0];
    for(int i = 1; i < grid.size(); i++) {
        dp[0][i] = dp[0][i-1] + grid[0][i];
    }
    for(int i = 1; i < grid.size(); i++) {
        dp[i][0] = dp[i-1][0] + grid[i][0];
    }
    for(int i = 1; i < row; i++) {
        for(int j = 1; j < row; j++) {
            dp[i][j] = std::min(dp[i][j-1] + grid[i][j],
                                dp[i-1][j] + grid[i][j]);
        }
    }
    return dp[row-1][row-1];
}

int rob(std::vector<int>& nums) {
  int col = nums.size();
  int dp[col];
  dp[0] = nums[0];
  dp[1] = nums[1];
  dp[2] = dp[0] + nums[2];
  for(int i = 3; i < col; i++) {
      dp[i] = std::max(dp[i-2], dp[i-3]) + nums[i];
  }
  return std::max(dp[col-2], dp[col-1]);
}

TreeNode* searchBST(TreeNode* root, int val) {
  while(root) {
      if(root->val == val)
        return root;
      if(root->val < val)
        root = root->right_child;
      if(root->val > val)
        root = root->left_child;      
  }
  return nullptr;
}

/*
          5
        3   7
       2 4 6 9

*/
// TreeNode* deleteNode(TreeNode* root, int key) {
//   TreeNode* will_delete_node = root;
//   TreeNode* will_delete_node_parent = root;
//   while(will_delete_node) {
//       if(will_delete_node->val == key) {
//           break;
//       }
//       else if(will_delete_node->val < key) {
//           will_delete_node_parent = will_delete_node_parent;
//           will_delete_node = will_delete_node->right_child;
//       } else {
//           will_delete_node_parent = will_delete_node;
//           will_delete_node = will_delete_node->left_child;
//       }
//   }
//   if(will_delete_node == nullptr)
//     return root;

//   if(will_delete_node->left_child == nullptr && will_delete_node->right_child == nullptr) {
//       if(will_delete_node_parent->val == will_delete_node->val)
//         return nullptr;
//       else if(will_delete_node_parent->val > will_delete_node->val);
//         will_delete_node_parent->left_child = nullptr;
//       else
//         will_delete_node_parent->right_child = nullptr;    
//       return root;
//   }
//   else if(will_delete_node->left_child == nullptr) {
//       if(will_delete_node_parent->val > will_delete_node->val)
//         will_delete_node_parent->left_child = will_delete_node->right_child;
//       else
//         will_delete_node_parent->right_child = will_delete_node->right_child;
//       return root;  
//   }
//   else if(will_delete_node->right_child == nullptr) {
//       if(will_delete_node_parent->val > will_delete_node->val)
//         will_delete_node_parent->left_child = will_delete_node->left_child;
//       else
//         will_delete_node_parent->right_child = will_delete_node->left_child;
//       return root;  
//   }
//   else {
//       TreeNode* max_in_left = will_delete_node->right_child;
//       while(max_in_left->right_child != nullptr) {
//           max_in_left = max_in_left->right_child;
//       }
//       TreeNode* min_in_right_parent = will_delete_node;
//       TreeNode* min_in_right = will_delete_node->right;
//       bool is1 = false;
//       while(min_in_right->left != nullptr) {
//           min_in_right_parent = min_in_right;
//           min_in_right = min_in_right->left;
//           is1 = true;
//       }
//       will_delete_node->val = min_in_right->val;
//       if(!is1)
//         min_in_right_parent->right = nullptr;
//       else
//         min_in_right_parent->left = nullptr;
//   }
//   return root;
// }

void TraverseTree(TreeNode* root, int& count) {
  if(root == nullptr)
    return;
  count += 1;
  TraverseTree(root->left_child, count);
  TraverseTree(root->right_child, count);  
}

int countNodes(TreeNode* root) {
  int count = 0;
  TraverseTree(root, count);
  return count;
}

TreeNode* pruneTree(TreeNode* root) {
    if(root == nullptr)
      return nullptr;
    root->left_child = pruneTree(root->left_child);
    root->right_child = pruneTree(root->right_child);
    if(root->left_child == nullptr && root->right_child == nullptr && root->val == 0) {
        return nullptr;
    }
    return root;
}

std::vector<int> maxSlidingWindow1(std::vector<int>& nums, int k) {
  std::deque<int> max_element_index;
  std::vector<int> results;
  for(int i = 0; i < nums.size(); i++) {
    //模拟向右滑动  
    if(!max_element_index.empty() && max_element_index.front() <= (i - k))
      max_element_index.pop_front();
    //从后往前弄掉比它小的元素  
    while(!max_element_index.empty() && nums[i] > nums[max_element_index.back()]) {
        max_element_index.pop_back();
    }
    max_element_index.push_back(i);
    if(i >= (k-1))
      results.push_back(nums[max_element_index.front()]);  
  }
  return results;
}

int lengthOfLongestSubstring(std::string s) {
  int max_result = 0;
  int i = 0;
  int j = 0;
  std::set<char> cs;
  while(i < s.size() && j < s.size()) {
      if(cs.count(s[j]) == 0) {
          cs.insert(s[j]);
          j++;
      } else {
          //重复出现的，不断删除，删除到j前面
          cs.erase(s[i]);
          i++;
      }
      max_result = std::max(max_result, j - i);
  }
  return max_result;
}

bool isMapSame(int* p_map, int* q_map) {
    for(int i = 0; i < 26; i++) {
        if(p_map[i] != q_map[i])
          return false;
    }
    return true;
}

std::vector<int> findAnagrams(std::string s, std::string p) {
  std::vector<int> result;
  int p_map[26], q_map[26];
  for(int i = 0; i < 26; i++) {
      p_map[i] = 0;
      q_map[i] = 0;
  }
  for(int i = 0; i < p.size(); i++) {
      p_map[p[i] - 'a']++;
  }
  int i = 0;
  int j = p.size() - 1;
  for(int i = 0; i < p.size(); i++) {
      q_map[s[i] - 'a']++;
  }
  while(j < s.size()) {
      if(isMapSame(p_map, q_map))
        result.push_back(i);
      q_map[s[i] - 'a']--;
      i++;
      j++;
      q_map[s[j] - 'a']++;
  }
  return result;
}

std::vector<std::vector<int>> levelOrderBottom(TreeNode* root) {
    std::vector<std::vector<int>> results;
    std::queue<TreeNode*> node_queue;
    if(root == nullptr)
      return results;
    node_queue.push(root);
    while(!node_queue.empty()) {
        //记录当前层的元素的个数
        int level_num = node_queue.size();
        std::vector<int> result;
        for(int i = 0; i < level_num; i++) {
            TreeNode* node = node_queue.front();
            node_queue.pop();
            result.push_back(node->val);
            if(node->left_child)
              node_queue.push(node->left_child);
            if(node->right_child)
              node_queue.push(node->right_child);  
        }
        results.insert(results.begin(), result);
    }
    return results;
}

//根据异或值对两个不同的值进行分组,基于异或值找到mask
std::vector<int> singleNumbers(std::vector<int>& nums) {
  int k = 0;
  for(auto num : nums) {
      k ^= num;
  }
  int mask = 1;
  while((k & mask) == 0)
    mask = mask << 1;
  //mask只有1位为1
  int a = 0;
  int b = 0;
  for(auto num : nums) {
      if((num & mask) == 0)
        a ^= num;
      else
        b ^= num;
  }
  return std::vector<int>{a, b};
}

/* input:
   1 2 3 4
   5 6 7 8
   9 10 11 12

   output: 1 2 3 4 8 12 11 10 9 5 6 7

   7
   9
   6
   关键是把索引序号弄正确了就好
*/
std::vector<int> spiralOrder(std::vector<std::vector<int>>& matrix) {
  int rows = matrix.size();
  int cols = matrix.front().size();
  int levels = std::round(std::min(rows, cols) / 2.0);
  std::cout<< levels << std::endl;
  std::vector<int> results;
  for(int i = 0; i < levels; i++) {
    // top row of every level
    for(int j = i; j < cols - i; j++) {
        results.push_back(matrix[i][j]);
    }
    // right col of every level
    for(int j = i+1; j < rows - i - 1 && (rows - i - 1) > i; j++) {
        results.push_back(matrix[j][cols-1-i]);
    }
    // bottom row of every level
    for(int j = cols-i-1; j >= i && (rows - 1 - i) > i; j--) {
        results.push_back(matrix[rows-1-i][j]);
    }
    // left col of every level
    for(int j = rows - 2 - i; j >= i + 1 && i < (cols - 1- i); j--) {
        results.push_back(matrix[j][i]);
    }
  }
  return results;
}

/*
前序遍历 preorder = [3,9,20,15,7]
中序遍历 inorder = [9,3,15,20,7]
output:
    3
   / \
  9  20
    /  \
   15   7

[1, 2], [1, 2] 

[1,2,3], [3,2,1]
    1
   / \
  2
 /
 3  
*/

TreeNode* buildTree(std::vector<int>& preorder, std::vector<int>& inorder) {
  if(preorder.empty())
    return nullptr;
  if(preorder.size() == 1) {
      return new TreeNode(preorder[0]);
  }  
  TreeNode* root = new TreeNode(preorder[0]);
  int root_index_inorder = 0;
  for(int i = 0; i < inorder.size(); i++) {
      if(inorder[i] == preorder[0]) {
          root_index_inorder = i;
          break;
      }
  }
  std::vector<int> left_in_order;
  for(int i = 0; i < root_index_inorder; i++) {
    left_in_order.push_back(inorder[i]);
  }
  std::vector<int> right_in_order;
  for(int i = root_index_inorder+1; i < inorder.size(); i++) {
    right_in_order.push_back(inorder[i]);
  }
  std::vector<int> left_pre_order;
  std::vector<int> right_pre_order;
  if(root_index_inorder == 0) {
    right_pre_order = std::vector<int>(preorder.begin() + 1, preorder.end());
  } else {
    int last_elemetn_in_left = inorder[root_index_inorder -1];
  
    for(int i = 1; i < preorder.size(); i++) {
      if(preorder[i] == last_elemetn_in_left) {
          left_pre_order.insert(left_pre_order.end(), preorder.begin() + 1, preorder.begin() + i + 1);
          right_pre_order.insert(right_pre_order.end(), preorder.begin() + i + 1, preorder.end());
      }
    }
    root->left_child = buildTree(left_pre_order, left_in_order);
  }
  root->right_child = buildTree(right_pre_order, right_in_order);
  
  return root;
}

//走过的路的长度一致就相遇了, 通过节点的交换来保证相遇
ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
    ListNode* node1 = headA;
    ListNode* node2 = headB;
    while(node1 != node2) {
        if(node1 != nullptr)
          node1 = node1->next; // node1 may be nullptr
        else
          node1  = headB;
        if(node2 != nullptr)
          node2 = node2->next;
        else
          node2 = headA;   
    }
    return node1;
}

// a simple two-dimension dynamic programming
int maxValue(std::vector<std::vector<int>>& grid) {
  int rows = grid.size();
  int cols = grid.front().size();
  std::vector<std::vector<int>> values(rows, std::vector<int>(cols, 0));
  values[0][0] = grid[0][0];
  for(int i = 1; i < cols; i++) {
    values[0][i] = values[0][i-1] + grid[0][i];
  }
  for(int i = 1; i < rows; i++) {
      values[i][0] = values[i-1][0] + grid[i][0];
  }
  for(int i = 1; i < cols; i++) {
      for(int j = 1; j < rows; j++) {
          values[j][i] = std::max(values[j][i-1], values[j-1][i]) + grid[j][i];
      }
  }
  return values[rows-1][cols-1];
}

// [1 3 2 6 5] is correct post order
// 分离开左右， 然后递归的验证二叉搜索树的特性，左边的比根部小，右边的比根部大
bool verifyPostorder(std::vector<int>& postorder) {
  if(postorder.empty())
    return true;  
  std::vector<int> left_part;
  std::vector<int> right_part;
  int root_value = postorder.back();
  for(int i = 0; i < postorder.size()-1; i++) {
    if(postorder[i] < root_value)
      left_part.push_back(postorder[i]);
    else {
        right_part = std::vector<int>(postorder.begin() + i, postorder.end()-1);
        break;
    }  
  }
  for(int i = 0; i < right_part.size(); i++) {
      if(right_part[i] < root_value)
        return false;
  }
  return verifyPostorder(left_part) && verifyPostorder(right_part);
}

//找到合适的一行，开始向左向下滑动 (row++) (col--)
bool findNumberIn2DArray(std::vector<std::vector<int>>& matrix, int target) {
  if(matrix.empty() || matrix.front().empty())
    return false;
  int start_row = -1;
  int rows = matrix.size();
  int cols = matrix.front().size();
  for(int i = 0; i < rows; i++) {
      if(target < matrix[i][cols - 1]) {
        start_row = i;
        break;
      }
      if(target == matrix[i][cols-1])
        return true;
  }
  if(start_row < 0)
    return false;
  int start_col = cols-2;
  while(1) {
      if(start_row > (rows-1) || start_col < 0)
        return false;
      if(target == matrix[start_row][start_col])
        return true;
      else if(target < matrix[start_row][start_col])
        start_col--;
      else
        start_row++;
  }

  return false;  
}

std::vector<int> printNumbers(int n) {
  std::vector<int> results;
  for(int i = 0; i < std::pow(10, n); i++)
    results.push_back(i);
  return results;
}

//递归动态相结合 利用之前的计算结果
std::vector<double> twoSum(int n) {
  std::vector<double> p_1 = {0.16667,0.16667,0.16667,0.16667,0.16667,0.16667};
  if(n == 1)
    return p_1;
  // from (n-1) --> 6(n-1)
  std::vector<double> pn_1 = twoSum(n-1);  
  int nums = n * 6 - n + 1;
  std::vector<double> result(nums, 0);
  for(int i = n; i < 6 * n + 1; i++) {
      for(int j = 1; j <= 6; j++) {
        if((i - j) < (n-1) || (i - j -n + 1) >= pn_1.size())
          continue;
        result[i - n] += 1.0 / 6.0 * pn_1[(i-j) - (n-1)];
      }
  }
  return result;
}

// c++ fast pow
// 3.1^ 10
// 3.1 * 3,1^2  * 3,1^4 
double myPow2(double x, int n) {
    int i = n;
    double res = 1;
    while(i) {
        if(i & 1)
          res *= x;
        x *= x;
        i /= 2;
    }
    if(n < 0)
      return 1.0 / res;
    return res;  
}

int numWays(int n) {
  if(n == 1 || n == 0)
    return 1;
  if(n == 2)
    return 2;
  std::vector<int> results;  
  results.resize(n+1);
  results[1] = 1;
  results[2] = 2;
  for(int i = 3; i<=n; i++) {
    results[i] = results[i-1] % 1000000007 + results[i-2] % 1000000007;;
  }
  return (results[n] % 1000000007);
}

int findRepeatNumber(std::vector<int>& nums) {
  std::map<int, int> num_count;
  for(int i = 0; i < nums.size(); i++) {
      if(num_count.count(nums[i]) == 0)
        num_count[nums[i]] = 1;
      else
      {
          return nums[i];
      }  
  }
  return 0;
}

std::vector<int> reversePrint(ListNode* head) {
  std::vector<int> results;
  ListNode* p = head;
  while(p != nullptr) {
      results.push_back(p->val);
      p = p->next;
  }
  std::reverse(results.begin(), results.end());
  return results;
}


std::string reverseLeftWords(std::string s, int n) {
  n = n % s.size();
  std::string part1 = s.substr(n, s.size() - n);
  std::string part2 = s.substr(0, n);
  std::cout << part1 << " " << part2 << std::endl;
  return part1 + part2;
}

int lengthOfLongestSubstring1(std::string s) {
  if(s.empty())
    return 0;
  int start = 0;
  int end = 1;
  std::vector<char> contained_chars;
  int max_length = 1;
  contained_chars.push_back(s[start]);
  while(end < s.size()) {
    for(int i = 0; i < contained_chars.size(); i++) {
        if(s[end] == s[i]) {
            start += (i + 1);
            std::cout<< start << std::endl;
            contained_chars.erase(contained_chars.begin(), contained_chars.begin() + i + 1);
            break;
        }
    }
    contained_chars.push_back(s[end]);
    max_length = std::max(max_length, end - start + 1);
    end++;
  }
  return max_length;
}

//动态规划，整数切分
int cuttingRope(int n) {
  std::vector<int> dp;
  dp.resize(n+1);
  dp[0] = 0;
  dp[1] = 0;  
  // dp[i]表示以i切分的最大的乘积
  for(int i = 2; i <= n; i++) {
      // split, j must larger than 0
      for(int j = 1; j < i; j++) {
        int tmp_result = std::max(j * (i - j), j * dp[i - j]);  
        dp[i] = std::max(dp[i], tmp_result);
      }
  }
  return dp[n];
}

/*
      1
    2   3
   4  5 
*/
/*
                     3
             4                5
        -7       -6  
    -7        -5
            -4 
*/

int sumOfLeftLeaves(TreeNode* root) {
//   int result = 0;
//   if(root == nullptr)
//     return result;
//   TreeNode* left = root->left_child;
//   if(left == nullptr)
//     return result; // left为空的时候，right不一定为空
//   if(left->left_child == nullptr && left->right_child == nullptr)
//     return left->val + sumOfLeftLeaves(root->right_child);
//   return sumOfLeftLeaves(left) + sumOfLeftLeaves(root->right_child);

  if(root == nullptr)
    return 0;
  TreeNode* left = root->left_child;
  TreeNode* right = root->right_child;
  if(left == nullptr)
    return sumOfLeftLeaves(right);
  if(left->left_child == nullptr && left->right_child == nullptr)
    return left->val + sumOfLeftLeaves(right);
  return sumOfLeftLeaves(left) + sumOfLeftLeaves(right);
}

std::vector<int> topKFrequent(std::vector<int>& nums, int k) {
  // key is number and value is count
  std::map<int, int> number_counts;
  for(int i = 0; i < nums.size(); i++) {
      if(number_counts.count(nums[i]) == 0)
        number_counts[nums[i]] = 1;
      else
      {
          number_counts[nums[i]]++;
      }  
  }
  //map 向vector转换以便进行排序
  std::vector<std::pair<int,int>> vector_number_counts;
  for(auto iter = number_counts.begin(); iter != number_counts.end(); iter++) {
      vector_number_counts.push_back({iter->first, iter->second});
  }
  std::sort(vector_number_counts.begin(), vector_number_counts.end(), 
  [](std::pair<int,int> p1, std::pair<int,int> p2) {
    return p1.second > p2.second;
  });
  std::vector<int> result;
  for(int i = 0; i< k;i++) {
    result.push_back(vector_number_counts[i].first);
  }
  return result;
}

//获取子结果，根据子结果推导入出最终结果
std::vector<std::vector<int>> subsets(std::vector<int>& nums) {
    std::vector<std::vector<int>> results;
    if(nums.empty())
      return results;
    if(nums.size() == 1) {
        results.push_back({nums.front()});
        results.push_back({});
        return results;
    }
    std::vector<int> sub_nums;
    for(int i = 0; i < nums.size() - 1; i++) {
        sub_nums.push_back(nums[i]);
    }
    std::vector<std::vector<int>> sub_results = subsets(sub_nums);
    for(int i = 0; i < sub_results.size(); i++) {
        results.push_back(sub_results[i]);
        std::vector<int> tmp = sub_results[i];
        tmp.push_back(nums.back());
        results.push_back(tmp);
    }
    return results;
}

//深度优先搜索，递归的进行颜色的涂改，涂改后颜色就不是初始颜色了
std::vector<std::vector<int>> floodFill(
    std::vector<std::vector<int>>& image, int sr, int sc, int newColor) {
  int rows = image.size();
  int cols = image.front().size();       
  int init_color = image[sr][sc];
  image[sr][sc] = newColor;
  if(newColor == init_color)
    return image;
  // same row
  if(image[sr][sc - 1] == init_color)
    floodFill(image, sr, sc-1, newColor);
  if(image[sr][sc + 1] == init_color)
    floodFill(image, sr, sc+1, newColor);
  // same col
  if(image[sr-1][sc] == init_color)
    floodFill(image, sr-1, sc, newColor);
  if(image[sr+1][sc] == init_color)
    floodFill(image, sr+1, sc, newColor);
  return image;
}

/*
输入：
[
  [0,2,1,0],
  [0,1,0,1],
  [1,1,0,1],
  [0,1,0,1]
]
输出： [1,2,4]
*/
int pondSizeHelper(std::vector<std::vector<bool>> is_visited, std::vector<std::vector<int>>& land, 
    int start_row, int start_col) {
  int pond_size = 0;
  if(is_visited[start_row][start_col])
    return 0;
  if(land[start_row][start_col] != 0) {
      is_visited[start_row][start_col] = true;
      return 0;
  } else {
      pond_size = 1;
  }  
  is_visited[start_row][start_col] = true;
  return pond_size + pondSizeHelper(is_visited, land, start_row, start_col + 1) + 
                     pondSizeHelper(is_visited, land, start_row+1, start_col) + 
                     pondSizeHelper(is_visited, land, start_row+1, start_col + 1);
}

std::vector<int> pondSizes(std::vector<std::vector<int>>& land) {
  std::vector<int> results;
  int rows = land.size();
  int cols = land.front().size();
  std::vector<std::vector<bool>> is_visited(rows, std::vector<bool>(cols, false));
  for(int i = 0; i < rows; i++) {
      for(int j = 0; j < cols; j++) {
          int result = pondSizeHelper(is_visited, land, i, j);
          if(result != 0)
            results.push_back(result);
      }
  }
  return results;
}

int search(std::vector<int>& nums, int target) {
  int count = 0;
  for(int i = 0; i < nums.size(); i++) {
      if(nums[i] == target)
        count++;
      if(nums[i] > target)
        break;  
  }
  return count;
}

/*
            3
          1    4
            2

*/
//左根右中序遍历
void zhongbianli(TreeNode* root, std::vector<int>& values) {
    if(root == nullptr)
      return;
    zhongbianli(root->left_child, values);
    values.push_back(root->val);
    zhongbianli(root->right_child, values);
}

int kthLargest(TreeNode* root, int k) {
  std::vector<int> values;
  zhongbianli(root, values);
  return values[values.size() - k];
}

int missingNumber(std::vector<int>& nums) {
  int n = nums.size();  
  int total_sum = n * (n + 1) / 2;
  int partsum = 0;
  for(auto e : nums) {
    partsum += e;
  }
  return total_sum - partsum;
}

std::vector<std::vector<int>> findContinuousSequence(int target) {
  std::vector<std::vector<int>> results;
  int half_num = target / 2;
  for(int i = 1; i <= half_num;i++) {
      int current_sum = i;
      int j = i+1;
      while(current_sum < target) {
        current_sum += j;
        j++;
      }
      if(current_sum == target) {
        std::vector<int> result;
        for(int k = i; k < j; k++) {
            result.push_back(k);
        }
        results.push_back(result);
      }
  }
  return results;
}

bool isStraight(std::vector<int>& nums) {
  std::vector<int> non_zero;
  for(int i = 0; i < nums.size(); i++) {
      if(nums[i] != 0)
        non_zero.push_back(nums[i]);
  }
  std::set<int> non_zero_set;
  for(int i = 0; i < non_zero.size(); i++) {
      non_zero_set.insert(non_zero[i]);
  }
  if(non_zero.size() != non_zero_set.size())
    return false;
  std::sort(non_zero.begin(), non_zero.end());
  
  // 非0的最大值和最小值的差距要小
  int count = non_zero.back() - non_zero.front() + 1;
  if(count <= 5)
    return true;

  return false;
}

/*
                  0                   0
              1        4 ==>      1        4
                2    3                 3
*/
int lastRemaining(int n, int m) {
  std::vector<int> nums;
  for(int i = 0; i < n; i++) {
      nums.push_back(i);
  }
  int index = 0;
  while(nums.size() > 1) {
    index = index % nums.size();
    if(index == (m - 1))
      nums.erase(nums.begin() + index);
    index++;  
  }
  return nums.front();
}

//小的元素pop后，只要大的元素还在就不会有影响
class MaxQueue {
public:
    MaxQueue() {

    }
    
    int max_value() {
      if(d.empty())
        return -1;
      return d.front();  
    }
    
    void push_back(int value) {
      q.push(value);
      while(!d.empty() && d.back() < value)
        d.pop_back();
      d.push_back(value);  
    }
    
    int pop_front() {
      if(q.empty())
        return -1;  
      int ans = q.front();
      if(ans == d.front())
        d.pop_front();
      q.pop();
      return ans;  
    }

    std::queue<int> q;
    //维护一个单调递减的双端队列
    std::deque<int> d;
};

int countOne(int num) {
    int count = 0;
    while(num != 0) {
        if((num % 10) == 1)
          count++;
        num /= 10;  
    }
    return count;
}

int countDigitOne1(int n) {
  int count = 0;
  for(int i = 0; i <= n; i++) {
      count += countOne(i);
  }
  return count;
}

//对于任意一个数字，个位，十位等等为1出现的次数
int countDigitOne(int n) {
  int result = 0;
  //分解低位，当前位，高位
  long low = 0;
  long high = n / 10;
  long cur = n % 10;
  long digit = 1;
  //每一位上是1的个数的相加总和
  while(high != 0 || cur != 0) {
      if(cur == 0) {
        result += high * digit;
      } else if(cur == 1) {
        result += high * digit + low + 1;
      } else {
          //cur == 2 - 9
        result += (high + 1) * digit;
      }
      low += cur * digit;
      cur = high % 10;
      high /= 10;
      digit *= 10;
  }
  return result;
}

//递增数组未旋转的话递增元素是第一个，旋转的话是第一个递减的元素
int minArray(std::vector<int>& numbers) {
  int n = numbers.size();
  if(n == 1)
    return numbers.front();
  for(int i = 0; i < numbers.size(); i++) {
    if(numbers[i] > numbers[i+1])
      return numbers[i+1];
  }
  return numbers.front();
}

//起点位置和字符深度遍历
bool exist_helper(std::vector<std::vector<char>>& board, 
                  std::vector<std::vector<bool>>& visited,
                  int i, int j, int k,
                  std::string word) {
  if(k >= word.size())
    return true;
  bool left = false;
  bool right = false;
  bool top = false;
  bool bottom = false;  
  if(j > 0 && visited[i][j-1] == false) {
      if(word[k] == board[i][j-1]) {
          visited[i][j-1] = true;
          left = exist_helper(board, visited, i, j-1, k+1,word);
          visited[i][j-1] = false;
      }
  }
  if(j < board.front().size()-1 && visited[i][j+1] == false) {
      if(word[k] == board[i][j+1]) {
          visited[i][j+1] = true;
          right = exist_helper(board, visited, i, j+1, k+1, word);
          visited[i][j+1] = false;
      }
  }
  if(i > 0) {
      if(word[k] == board[i-1][j] && visited[i-1][j] == false) {
          visited[i-1][j] = true;
          top = exist_helper(board, visited, i-1, j, k+1,word);
          visited[i-1][j] = false;
      }
  }
  if(i < board.size() - 1 && visited[i+1][j] == false) {
      if(word[k] == board[i+1][j]) {
          visited[i+1][j] = true;
          bottom = exist_helper(board, visited, i+1, j, k+1,word);
          visited[i+1][j] = false;
      }
  }
  return left || right || top || bottom;
}

bool exist(std::vector<std::vector<char>>& board, std::string word) {
  int rows = board.size();
  int cols = board.front().size();  
  std::vector<std::vector<bool>> visited(rows, std::vector<bool>(cols, false));
  int k = 1;
  for(int i = 0; i < rows; i++) {
      for(int j = 0; j < cols; j++) {
          if(board[i][j] == word.front()) {
              visited[i][j] = true;
              if(exist_helper(board, visited, i, j, k, word))
                return true;
              visited[i][j] = false;  
          }
      }
  }
  return false;
}

char firstUniqChar(std::string s) {
  if(s.empty())
    return ' ';
  std::unordered_map<char, bool> char_count;
  for(int i = 0; i < s.size(); i++) {
      if(char_count.count(s[i]) == 0)
        char_count.insert({s[i], true});
      else
      {
          char_count[s[i]] = false;
      }
  }
  //遍历字符串找到第一个而不是遍历哈希表
  for(int i = 0; i < s.size(); i++) {
      if(char_count[s[i]])
        return s[i];
  } 
  return s.front();
}

//找到要删除的节点，指针重新连接一下
ListNode* deleteNode(ListNode* head, int val) {
  if(head->val == val)
    return head->next;
  ListNode* p = head;
  ListNode* q = p->next;
  while(q && q->val != val) {
      p = p->next;
      q = q->next;
  }
  p->next = q->next;
  return head;  
}

//暴力解法
int maxProfit1(std::vector<int>& prices) {
  int max_profit = 0;
  for(int i = 0; i < prices.size()-1;i++) {
      for(int j = i+1;j < prices.size(); i++) {
          max_profit = std::max(max_profit, prices[j] - prices[i]);
      }
  }
  return max_profit;
}

//一维动态规划解法
int maxProfit(std::vector<int>& prices) {
    if(prices.size() < 2)
      return 0;
    int max_profit = 0;
    int min_price = prices[0];
    std::vector<int> dp(prices.size(), 0);
    //当日卖出价减去过去的最低价
    for(int i = 1; i < prices.size(); i++) {
        dp[i] = std::max(dp[i-1], prices[i] - min_price);
        min_price = std::min(min_price,prices[i]);
    }
    return dp.back();
}

class Student {

};

class Student1 {
public:
    virtual ~Student1() {}
};

void out_sizeof() {
    Student xiaoming;
    Student1 xiaowang;
    //输出1 ，实例要占据1个字节的内存
    std::cout << sizeof(xiaoming) << std::endl;
    //输出8， 有虚函数之后，每个实例会含有一个指向虚函数表的指针，64位机器上一个指针8个字节
    std::cout << sizeof(xiaowang) << std::endl; 
}

class D {
    private:
    int value;

    public:
    D(int n) {
        value = n;
    }

    //拷贝构造函数第一个参数必须是引用传递
    D(const D& other) {
        value = other.value;
    }

    void Print() {
        std::cout <<"value: " << value << std::endl;
    }
};


class MyString {
    private:
    char* m_pData;

    public:
    MyString(char* pData = nullptr) {
        if(pData == nullptr) {
            m_pData = new char[1];
            m_pData[0] = '\0';
        }
        int length = strlen(pData);
        //分配内存，再进行拷贝
        m_pData = new char[length + 1];
        strcpy(m_pData, pData);
    }

    MyString(const MyString& str) {
      int length = strlen(str.m_pData);
      m_pData = new char[length + 1];
      strcpy(m_pData, str.m_pData);
    }
    ~MyString() {
        delete []m_pData;
        m_pData = nullptr;
    }

    MyString& operator=(const MyString& str) {
        //防止自赋值
        if(this == &str)
          return *this;

        delete []m_pData;
        m_pData = nullptr;
        m_pData = new char[strlen(str.m_pData) + 1];
        strcpy(m_pData, str.m_pData);
        //返回引用
        return *this;  
    }

    void Print() {
        printf("%s", m_pData);
    }

};

class Singleton {
    private:
    static Singleton* m_instance;
    std::mutex mtx;
    Singleton() {}
    Singleton(const Singleton& ton) = delete;
    Singleton& operator=(const Singleton& ton) = delete;

    public:
    Singleton* GetInstance() {
        if(m_instance == nullptr) {
            mtx.lock();
            if(m_instance == nullptr)
                m_instance = new Singleton(); //只会有一个线程进入并创建，另一个直接返回
            mtx.unlock();
        }
        return m_instance;  
    }
};

Singleton* Singleton::m_instance = nullptr;

std::string replaceSpace(std::string s) {
  // we are happy
  std::string result;
  //string 可以直接push_back,类似vector
  for(auto c : s) {
      if(c == ' ') {
          result.push_back('%');
          result.push_back('2');
          result.push_back('0');
      } else
      {
          result.push_back(c);
      }
  }
  return result;
}

bool is_index_valid(int row, int col, int k) {
    int result = 0;
    result += row / 10;
    result += row % 10;
    result += col / 10;
    result += col % 10;
    return result <= k;
}

int dfs(int i, int j, int m, int n, int k, std::vector<std::vector<bool>>& visited) {
  //访问过的或者无效的不进行累加
  if(i < 0 || i >= m || j < 0 || j >= n || visited[i][j] ||!is_index_valid(i, j, k))
    return 0;
  visited[i][j] = true;
  //当前值加上四条路径的累加值
  return 1 +  dfs(i, j+1, m, n, k, visited) + dfs(i+1, j, m, n, k, visited) + 
  dfs(i, j-1, m, n, k, visited) + dfs(i-1, j, m, n, k, visited);
}

int movingCount(int m, int n, int k) {
    std::vector<std::vector<bool>> is_visited(m, std::vector<bool>(n, false));
    return dfs(0, 0, m ,n, k, is_visited);
}

std::vector<int> exchange(std::vector<int>& nums) {
  // 奇数在前，偶数在后
  int i = 0;
  int j = nums.size() - 1;
  while(i < j) {
      if(nums[i] % 2 == 0 && nums[j] % 2 == 1) {
          std::swap(nums[i], nums[j]);
          i++;
          j--;
      } else if(nums[i] % 2 == 0 && nums[j] % 2 == 0) {
          j--;
      } else if(nums[i] % 2 == 1 && nums[j] % 2 == 0) {
          i++;
          j--;
      } else {
          i++;
      }
  }
  return nums;
}

class Parent {
    public:
    virtual void print() {
        std::cout<<"parent" << std::endl;
    }
    virtual void print_hello() {}
    int value;
};

class child1 : public Parent {
   public:
   virtual void print() override {
     std::cout<<"child1 " << std::endl;
   }
   int value;
};

class child2 : public Parent{
   public:
   int value;
};

ListNode* getKthFromEnd(ListNode* head, int k) {
  if(head == nullptr)
    return nullptr;
  ListNode* p = head;
  ListNode* q = p;
  for(int i = 0; i < k; i++) {
      q = q->next;
  }
  while(q != nullptr) {
      p = p->next;
      q = q->next;
  }
  return p;
}

//1-2-4 1-3-4 --》1 1 2 3 4 4
ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
  if(l1 == nullptr)
    return l2;
  if(l2 == nullptr)
    return l1;
  ListNode* p = l1;  
  ListNode* q = l2;
  ListNode* k;
  if(p->val < q->val) {
      k = p;
      p = p->next;
  } else {
      k = q;
      q = q->next;
  }
  //建立一个新的指针
  ListNode* result = k;
  while(p && q) {
      if(p->val < q->val) {
          k->next = p;
          p = p->next;
      } else {
          k->next = q;
          q = q->next;
      }
      k = k->next;
  }
  if(p)
    k->next = p;
  else
    k->next = q;
  return result;
}

//判断弹出序列是否满足条件
bool validateStackSequences(std::vector<int>& pushed, std::vector<int>& popped) {
  //用一个辅助栈模拟推入的过程，值相等则进行推出
  std::stack<int> fuzhu;
  int j = 0;
  int i = 0;
  while(i < pushed.size()) {
      fuzhu.push(pushed[i]);
      while(!fuzhu.empty() && fuzhu.top() == popped[j]) {
          fuzhu.pop();
          j++;
      }
      i++;
  }
  //依次的pop
  while(j < popped.size()) {
      if(fuzhu.top() == popped[j]) {
          fuzhu.pop();
          j++;
      } else {
          return false;
      }
  }
  return fuzhu.empty();
}

/*
输入: n = 10
输出: 12
解释: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12 是前 10 个丑数。
*/
int nthUglyNumber(int n) {
  std::vector<int> ugly_nums(n, 1);
  int a = 0, b = 0, c = 0;
  for(int i = 1; i < n; i++) {
      int n1 = 2 * ugly_nums[a];
      int n2 = 3 * ugly_nums[b];
      int n3 = 5 * ugly_nums[c];
      ugly_nums[i] = std::min(n1, std::min(n2, n3));
      if(ugly_nums[i] == n1)
        a++;
      if(ugly_nums[i] == n2)
        b++;
      if(ugly_nums[i] == n3)
        c++;    
  }
  return ugly_nums.back();
}

std::vector<int> getLeastNumbers(std::vector<int>& arr, int k) {
  if(k <= 0)
    return {};
  std::priority_queue<int> arr_k;
  for(int i = 0; i < arr.size();i++) {
      if(i < k) {
          arr_k.push(arr[i]);
      } else {
          if(arr[i] < arr_k.top()) {
              arr_k.pop();
              arr_k.push(arr[i]);
          }
      }
  }
  std::vector<int> result;
  while(!arr_k.empty()) {
      result.push_back(arr_k.top());
      arr_k.pop();
  }
  return result;
}

class Node {
public:
    int val;
    Node* next;
    Node* random;
    
    Node(int _val) {
        val = _val;
        next = NULL;
        random = NULL;
    }
};

Node* copyRandomList(Node* head) {
    return nullptr;
}

//构建乘积数组，上三角和下三角
/*
  b[0] 1 a[1] a[2] a[3] a[4] a[5]
  b[1] a[0] 1 a[2] a[3] a[4] a[5]
  b[2] a[0] a[1] 1 a[3] a[4] a[5] 
  b[3] a[0] a[1] a[2] 1 a[4] a[5]
  b[4] a[0] a[1] a[2] a[3] 1 a[5]
  b[5] a[0] a[1] a[2] a[3] a[4] 1

  分为两个三角区域，递推式的进行推算
*/
std::vector<int> constructArr(std::vector<int>& a) {
  std::vector<int> b(a.size(), 1);
  for(int i = 1; i < a.size(); i++) {
      b[i] = b[i-1] * a[i-1];
  }
  int tmp = 1;
  for(int i = a.size() - 2; i >= 0; i--) {
      tmp *= a[i+1];
      b[i] *= tmp;
  }
  return b;
}

//约瑟夫回环问题
// f(n, m) = (f(n - 1, m) + m) % n
int lastRemaining1(int n, int m) {
    if(n == 1)
      return 0;
    int result = 0;  
    for(int i = 2; i <= n; i++) {
      result = (result + m) % i;
    }
    return result;
}

//左右子树交换，然后递归的进行交换
TreeNode* mirrorTree(TreeNode* root) {
  if(root == nullptr)
    return nullptr;
  TreeNode* tmp = root->left_child;
  root->left_child = root->right_child;
  root->right_child = tmp;
  mirrorTree(root->left_child);
  mirrorTree(root->right_child);
  return root;  
}

class MedianFinder {
public:
    /** initialize your data structure here. */
    MedianFinder() {

    }
    
    //每次都在有序数组中插入元素
    void addNum(int num) {
      if(nums.empty()) {
        nums.push_back(num);
      } else {
        auto iter = std::upper_bound(nums.begin(), nums.end(), num);
        if(iter == nums.end())
          nums.push_back(num);
        else {
          nums.insert(iter, num);  
        }  
      }
    }
    
    double findMedian() {
      if(nums.size() % 2 == 1) {
          return nums[(nums.size() - 1) / 2];
      } else {
          return (nums[nums.size() / 2 - 1] + nums[nums.size() / 2]) / 2.0;
      }
    }

    std::vector<int> nums;
};

void Erase() {
    std::vector<int> a;
    for(int i = 1; i<=10; i++) {
        a.push_back(i);
    }
    for(auto iter = a.begin(); iter != a.end();) {
        if((*iter) % 2 == 0) {
            iter = a.erase(iter);
        } else {
            iter++;
        }
    }
    for(auto e : a ) {
        std::cout<<e <<" ";
    }
    std::cout<<std::endl;
}

int searchInsert(std::vector<int>& nums, int target) {
  int start_index = 0;
  int end_index = nums.size() - 1;
  int middle_index = start_index + (end_index - start_index) / 2;
  while(start_index < end_index) {
      if(nums[middle_index] == target)
        return middle_index;
      else if(nums[middle_index] < target) {
          start_index = middle_index + 1;
      } else {
          end_index = middle_index - 1;
      }
      middle_index = start_index + (end_index - start_index) / 2;
  }
  if(target > nums[start_index])
    return start_index + 1;
  else
    return start_index;  
}

std::vector<int> sortedSquares(std::vector<int>& A) {
  std::vector<int> less_0;
  std::vector<int> more_0;
  for(int i = 0; i < A.size();i++) {
      if(A[i] < 0)
        less_0.push_back(A[i]);
      else
        more_0.push_back(A[i]);
  }
  std::vector<int> results;
  int i = 0;
  int j = less_0.size() - 1;
  while(i < more_0.size() && j >= 0) {
      if(abs(less_0[j]) > more_0[i]){
          results.push_back(more_0[i] * more_0[i]);
          i++;
      } else {
          results.push_back(less_0[j] * less_0[j]);
          j--;
      }
  }
  while(i < more_0.size()) {
      results.push_back(more_0[i] * more_0[i]);
      i++;
  }
  while(j >=0) {
      results.push_back(less_0[j] * less_0[j]);
      j--;
  }
  return results;
}

int minSubArrayLen(int s, std::vector<int>& nums) {
  int result = nums.size();
  bool exist = false;
  for(int i = 0; i < nums.size(); i++) {
      int sum = 0;
      for(int j = i; j < nums.size(); j++) {
        sum += nums[j];
        if(sum >= s) {
            result = std::min(result, (j - i + 1));
            exist = true;
            break;
        }
      }
  }
  if(!exist)
    return 0;
  return result;
}

/*
      1  2  3  4     1 2 3
     12  13 14 5     8 9 4
     11  16 15 6     7 6 5
     10  9  8  7
*/

std::vector<std::vector<int>> generateMatrix(int n) {
  int circle_count = std::round(n / 2.0);
  std::vector<std::vector<int>> matrix(n, std::vector<int>(n, 0));
  int start_top_num = 1;
  for(int i = 0; i < circle_count; i++) {
      // top row: i, col: i - (n-i)
      for(int j = i; j < (n - i) ; j++) {
          matrix[i][j] = start_top_num + (j - i);
      }
      start_top_num += (4 * (n - 2 * i)) - 4;

      //right col: n-i-1 row: i+1 - n-i-2
      for(int j = (i+1); j < (n-i-1); j++) {
          matrix[j][n-i-1] = matrix[i][n-i-1] + (j - i);
      }

      for(int j = n - 1-i; j >= i ; j--) {
          if(n % 2 == 1 && i == (circle_count - 1))
            continue;
          matrix[n-i-1][j] = matrix[n-i-2][n-i-1] + (n - i - j);
      }

      //left col: i row: n-i-2 i+1
      for(int j = (n-2-i); j > i;j--) {
          matrix[j][i] = matrix[n-1-i][i] + (n-1-i-j);
      }
  }
  return matrix;
}

ListNode* removeElements(ListNode* head, int val) {
  if(head == nullptr)
    return nullptr;
  ListNode* sential = new ListNode(0);
  sential->next = head;
  ListNode* pre = sential;
  ListNode* cur = head;
  ListNode* next = head->next;
  while(cur) {
      std::cout<<cur->val<<std::endl;
      if(cur->val == val) {
          pre->next = next;
      } else {
          pre = cur;
      }
      cur = cur->next;
      if(cur == nullptr)
        break;
      next = cur->next;
  }
  return sential->next;
}

class MyLinkedList {
public:
    /** Initialize your data structure here. */
    MyLinkedList() {

    }
    
    /** Get the value of the index-th node in the linked list. If the index is invalid, return -1. */
    int get(int index) {
      if(index < 0 || index >= nums.size())
        return -1;
      auto iter = nums.begin();
      std::advance(iter, index);
      return *iter;  
    }
    
    /** Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list. */
    void addAtHead(int val) {
      nums.push_front(val);
    }
    
    /** Append a node of value val to the last element of the linked list. */
    void addAtTail(int val) {
      nums.push_back(val);
    }
    
    /** Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted. */
    void addAtIndex(int index, int val) {
      if(index < 0 || index > nums.size())
        return;
      auto iter = nums.begin();
      std::advance(iter, index);  
      nums.insert(iter, val); 
    }
    
    /** Delete the index-th node in the linked list, if the index is valid. */
    void deleteAtIndex(int index) {
      if(index < 0 || index >= nums.size())
        return;
      auto iter = nums.begin();
      std::advance(iter, index);  
      nums.erase(iter);
    }

    std::list<int> nums;
    
};

//快慢指针 f - s = nb, s = nb, 环形如口处在a+nb处，相遇的时候让慢指针在走a步即可
ListNode *detectCycle(ListNode *head) {
  ListNode* fast = head;
  ListNode* slow = head;
  while(fast) {
    if(fast->next != nullptr)  
      fast = fast->next->next;
    else
      return nullptr;  
    slow = slow->next;
    if(fast == slow)
      break;  
  }
  if(fast == nullptr)
    return nullptr; 
  fast = head;
  while(fast) {
    if(fast == slow)
      return fast;
    fast = fast->next;
    slow = slow->next;
  }
  return nullptr;
}

bool canConstruct(std::string ransomNote, std::string magazine) {
  std::unordered_map<char, int> char_counts;
  for(int i = 0; i < magazine.size(); i++) {
      if(char_counts.count(magazine[i]) == 0)
        char_counts[magazine[i]] = 1;
      else
      {
          char_counts[magazine[i]]++;
      }   
  }
  for(int i = 0; i < ransomNote.size(); i++) {
      if(char_counts.count(ransomNote[i]) == 0)
        return false;
      if(char_counts[ransomNote[i]] <= 0)
        return false;
      char_counts[ransomNote[i]]--;
  }
  return true;
}

int distributeCandies(std::vector<int>& candies) {
  int kinds = 1;
  std::sort(candies.begin(), candies.end());
  int pre_kind = candies.front();
  for(int i = 1; i < candies.size(); i++) {
    if(candies[i] == pre_kind)
      continue;
    else {
        kinds++;
        pre_kind = candies[i];
    }  
  }
  if(kinds < candies.size() / 2)
    return kinds;
  else
    return candies.size() / 2;  
}

int get_square_sum(int n) {
    int sum = 0;
    while(n) {
      int num = n % 10;
      n /= 10;
      sum += num * num;
    }
    return sum;
}

// 通过hashset检测是否会出现循环
bool isHappy(int n) {
  std::map<int, int> count;
  count[n] = 1;  
  while(n) {
      if(n == 1)
        return true;
      n = get_square_sum(n);
      if(count.count(n) != 0)
        return false;
      else
        count[n] = 1; 
  }
  return false;
}

//用hash降低一半的复杂度
int fourSumCount1(std::vector<int>& A, std::vector<int>& B, 
                 std::vector<int>& C, std::vector<int>& D) {
  //ab之和能够达到某一个value的有多少个数字                   
  std::unordered_map<int, int> AB;
  for(int i = 0;i < A.size(); i++) {
      for(int j = 0; j < B.size(); j++) {
          int sum = A[i] + B[j];
          if(AB.count(sum) == 0)
            AB[sum] = 1;
          else
            AB[sum]++;  
      }
  }

  int result = 0;
  for(int i = 0;i < C.size(); i++) {
      for(int j = 0; j < D.size();j++) {
          int sum = C[i] + D[j];
          if(AB.count(-sum) != 0) {
              result += AB[-sum];
          }
      }
  }
  return result;
}

bool containsNearbyDuplicate(std::vector<int>& nums, int k) {
  std::unordered_map<int, int> num_index;
  for(int i = 0; i < nums.size();i++) {
      if(num_index.count(nums[i]) == 0)
        num_index[nums[i]] = i;
      else {
          if((i - num_index[nums[i]]) <= k)
            return true;
          else
            num_index[nums[i]] = i;  
      }  
  }
  return false;
}

int strStr(std::string haystack, std::string needle) {
  if(needle.empty())
    return 0;
  if(needle.size() > haystack.size())
    return -1;  
  for(int i = 0; i < haystack.size() - needle.size() + 1;i++) {
      std::string tmp = haystack.substr(i,needle.size());
      if(tmp == needle)
        return i;
  }
  return 0;  
}

bool isPattern(std::string a, std::string b) {
    int start_index = 0;
    int end_index = a.size() - 1;
    bool flag = false;
    while(end_index < b.size()) {
        std::string tmp = b.substr(start_index, a.size());
        if(tmp != a)
          return false;
        start_index += a.size();
        end_index += a.size();
        flag = true;
    }
    return flag;
}

//提取子串，不断进行重复判断
bool repeatedSubstringPattern(std::string s) {
  for(int i = 0; i < s.size() / 2; i++) {
    if(s.size() % (i+1) != 0)
      continue;  
    std::string sub_str = s.substr(0, i+1);
    if(isPattern(sub_str, s))
      return true;
  }
  return false;
}

class MyQueue {
public:
    /** Initialize your data structure here. */
    MyQueue() {

    }
    
    /** Push element x to the back of queue. */
    void push(int x) {
      if(a.empty())
        front_element = x;  
      a.push(x);
    }
    
    /** Removes the element from in front of queue and returns that element. */
    int pop() {
      while(!a.empty()) {
          b.push(a.top());
          a.pop();
      }
      int result = b.top();
      b.pop();
      if(!b.empty())
        front_element = b.top();
      while(!b.empty()) {
          a.push(b.top());
          b.pop();
      }
      return result;
    }
    
    /** Get the front element. */
    int peek() {
      return front_element;
    }
    
    /** Returns whether the queue is empty. */
    bool empty() {
      return a.empty();
    }

    std::stack<int> a;
    std::stack<int> b;

    int front_element;
};

class MyStack {
public:
    /** Initialize your data structure here. */
    MyStack() {

    }
    
    /** Push element x onto stack. */
    void push(int x) {
      a.push(x);
      top_element = x;
    }
    
    /** Removes the element on top of the stack and returns that element. */
    int pop() {
      int result = 0;  
      while(!a.empty()) {
          if(a.size() == 1) {
              result = a.front();
              a.pop();
          } else {
            b.push(a.front());
            a.pop();
          }
      }
      while(!b.empty()) {
          if(b.size() == 1)
            top_element = b.front();
          a.push(b.front());
          b.pop();
      }
      return result;
    }
    
    /** Get the top element. */
    int top() {
      return top_element;
    }
    
    /** Returns whether the stack is empty. */
    bool empty() {
      return a.empty();
    }

    std::queue<int> a;
    std::queue<int> b;
    int top_element;
};

bool isValid(std::string s) {
  std::stack<char> cs;
  for(int i = 0; i < s.size(); i++) {
      if(cs.empty())
        cs.push(s[i]);
      else {
          if(s[i] == '(' || s[i] == '[' || s[i] == '{')
            cs.push(s[i]);
          else if(s[i] == ')') {
              if(cs.top() != '(')
                return false;
              else
                cs.pop();  
          }
          else if(s[i] == ']') {
              if(cs.top() != '[')
                return false;
              else
                cs.pop();     
          }
          else {
              if(cs.top() != '{')
                return false;
              else
                cs.pop();  
          }
      }  
  }
  return cs.empty();
}

//use stack structure to remove string duplicates
std::string removeDuplicates(std::string S) {
  std::stack<char> cs;
  for(int i = 0; i < S.size(); i++) {
      if(cs.empty())
        cs.push(S[i]);
      else {
          if(S[i] == cs.top())
            cs.pop();
          else
            cs.push(S[i]);  
      }  
  }
  std::string a;
  while(!cs.empty()) {
    a.push_back(cs.top());
    cs.pop();
  }
  std::reverse(a.begin(), a.end());
  return a;
}

int largestRectangleArea(std::vector<int>& heights) {
    int n = heights.size();
    std::vector<int> lower(n, 0);
    std::vector<int> higher(n, 0);
    for(int i=0; i < heights.size(); i++) {
      int j1 = i-1;
      while(j1 >=0) {
          if(heights[j1] < heights[i]) {
              lower[i] = j1;
              break;
          } else {
              j1--;
          }
      }
      if(j1 < 0)
        lower[i] = -1;

      int j = i+1;
      while(j < heights.size()) {
          if(heights[j] < heights[i]) {
              higher[i] = j;
              break;
          } else {
              j++;
          }
      }
      if(j == heights.size())
        higher[i] = heights.size();  
    }

    int result = 0;
    for(int i = 0; i < heights.size(); i++) {
        result = std::max((higher[i] - lower[i] - 1) * heights[i], result);
    }
    return result;
}

class Node_N {
public:
    int val;
    std::vector<Node_N*> children;

    Node_N() {}

    Node_N(int _val) {
        val = _val;
    }

    Node_N(int _val, std::vector<Node_N*> _children) {
        val = _val;
        children = _children;
    }
};

void preorder1(Node_N* node, std::vector<int>& nums) {
    if(node == nullptr)
      return;
    nums.push_back(node->val);
    for(int i = 0; i < node->children.size(); i++) {
        preorder1(node->children[i], nums);
    }  
}

std::vector<int> preorder(Node_N* root) {
    std::vector<int> result;
    preorder1(root, result);
    return result;
}

void postorder1(Node_N* node, std::vector<int>& nums) {
    if(node == nullptr)
      return;
    for(int i = 0; i < node->children.size(); i++) {
        postorder1(node->children[i], nums);
    }
    nums.push_back(node->val);
}

std::vector<int> postorder(Node_N* root) {
    std::vector<int> result;
    postorder1(root, result);
    return result;
}

std::vector<int> rightSideView(TreeNode* root) {
  std::vector<int> result;
  std::queue<TreeNode*> nodes;
  if(root == nullptr)
    return result;
  nodes.push(root);
  std::vector<TreeNode*> current_level_nodes;
  current_level_nodes.push_back(root);
  while(!nodes.empty()) {
    while(!nodes.empty()) {
      current_level_nodes.push_back(nodes.front());
      nodes.pop();
    }
    result.push_back(current_level_nodes.back()->val);
    for(int i = 0; i < current_level_nodes.size(); i++) {
        if(current_level_nodes[i]->left_child != nullptr)
          nodes.push(current_level_nodes[i]->left_child);
        if(current_level_nodes[i]->right_child != nullptr)
          nodes.push(current_level_nodes[i]->right_child);  
    }
    current_level_nodes.clear();
  }
  return result;
}

std::vector<double> averageOfLevels(TreeNode* root) {
  std::vector<double> result;
  std::queue<TreeNode*> nodes;
  if(root == nullptr)
    return result;
  nodes.push(root);
  std::vector<TreeNode*> current_level_nodes;
  while(!nodes.empty()) {
      while(!nodes.empty()) {
          current_level_nodes.push_back(nodes.front());
          nodes.pop();
      }
      double sum = 0;
      for(int i = 0; i < current_level_nodes.size(); i++) {
          sum += current_level_nodes[i]->val;
      }
      result.push_back(sum / current_level_nodes.size());
      for(int i = 0; i < current_level_nodes.size(); i++) {
          if(current_level_nodes[i]->left_child != nullptr)
            nodes.push(current_level_nodes[i]->left_child);
          if(current_level_nodes[i]->right_child != nullptr)
            nodes.push(current_level_nodes[i]->right_child);  
      }
      current_level_nodes.clear();
  }
  return result;  
}

TreeNode* invertTree(TreeNode* root) {
  if(root == nullptr)
    return nullptr;
  if(root->left_child == nullptr && root->right_child == nullptr)
    return nullptr;
  TreeNode* tmp = root->left_child;
  root->left_child = root->right_child;
  root->right_child = tmp;
  invertTree(root->left_child);
  invertTree(root->right_child);
  return root;
}

//先填满当前层，再把下一层的数据都放到队列中
int findBottomLeftValue(TreeNode* root) {
  int result = 0;
  std::queue<TreeNode*> nodes;
  if(root == nullptr)
    return result;
  nodes.push(root);
  std::vector<TreeNode*> current_level_nodes;
  current_level_nodes.push_back(root);
  while(!nodes.empty()) {
    while(!nodes.empty()) {
      current_level_nodes.push_back(nodes.front());
      nodes.pop();
    }
    result = current_level_nodes.front()->val;
    for(int i = 0; i < current_level_nodes.size(); i++) {
        if(current_level_nodes[i]->left_child != nullptr)
          nodes.push(current_level_nodes[i]->left_child);
        if(current_level_nodes[i]->right_child != nullptr)
          nodes.push(current_level_nodes[i]->right_child);  
    }
    current_level_nodes.clear();
  }
  return result;
}

bool hasPathSum(TreeNode* root, int sum) {
    if(root == nullptr)
      return false;
    if(root->left_child == nullptr && root->right_child == nullptr) {
        if(root->val == sum)
          return true;
        else
          return false;
    }
    return hasPathSum(root->left_child, sum - root->val) || hasPathSum(root->right_child, sum - root->val);  
}

//中间的result参数代表了当前遍历的结果
void binaryTreePaths_helper(TreeNode* root, std::vector<int> result, std::vector<std::vector<int>>& results) {
    if(root == nullptr)
      return;
    result.push_back(root->val);
    if(root->left_child == nullptr && root->right_child == nullptr) {
        results.push_back(result);
    } else {
        binaryTreePaths_helper(root->left_child, result, results);
        binaryTreePaths_helper(root->right_child, result, results);
    }
}

std::vector<std::string> binaryTreePaths(TreeNode* root) {
  std::vector<std::string> result_strings;
  std::vector<int> result;
  std::vector<std::vector<int>> results;
  binaryTreePaths_helper(root, result, results);
  for(int i = 0;i < results.size(); i++) {
      std::string tmp = std::to_string(results[i].front());
      for(int j = 1; j < results[i].size(); j++) {
          tmp += ("->" + std::to_string(results[i][j]));
      }
      result_strings.push_back(tmp);
  }
  return result_strings;
}

std::vector<std::vector<int>> pathSum(TreeNode* root, int sum) {
  std::vector<int> result;
  std::vector<std::vector<int>> results;
  binaryTreePaths_helper(root, result, results);
  std::vector<std::vector<int>> final_results;
  for(int i = 0; i < results.size(); i++){
    int tmp_sum = 0;
    for(int j = 0; j < results[i].size(); j++) {
        tmp_sum += results[i][j];
    }
    if(tmp_sum == sum)
      final_results.push_back(results[i]);
  }
  return final_results;
}

TreeNode* mergeTrees(TreeNode* t1, TreeNode* t2) {
  if(t1 == nullptr)
    return t2;
  if(t2 == nullptr)
    return t1;
  t1->val += t2->val;
  t1->left_child = mergeTrees(t1->left_child, t2->left_child);
  t1->right_child = mergeTrees(t1->right_child, t2->right_child);
  return t1;    
}

//尽可能的让饥饿度最小的孩子先满足需求
int findContentChildren(std::vector<int>& g, std::vector<int>& s) {
  std::sort(g.begin(), g.end());
  std::sort(s.begin(), s.end());
  int result = 0;
  int i = 0;
  int j = 0;
  while(i < g.size() && j < s.size()) {
      if(g[i] <= s[j]) {
          result++;
          i++;
          j++;
      } else {
          j++;
      }
  }
  return result;
}

//初始全部置为1，从左到右遍历一遍，比左边评分大，则糖果加1
int candy(std::vector<int>& ratings) {
  std::vector<int> candies(ratings.size(), 1);
  for(int i = 1; i < ratings.size(); i++) {
      if(ratings[i] > ratings[i-1])
        candies[i] = candies[i-1] + 1;
  }
  //从右到左遍历，比右边大，则对应的糖果比右边多1
  for(int i = ratings.size()-2; i >= 0; i--) {
      if(ratings[i] > ratings[i+1]) {
          if(candies[i] <= candies[i+1])
            candies[i] = candies[i+1] + 1;
      }
  }
  return std::accumulate(candies.begin(), candies.end(), 0);
}

void delete_test() {
    int* a = new int(100);
    int* b = a;
    for(int i = 0; i< 100;i++) {
        a[i] = i;
    }
    a = nullptr;
    //delete []a;
    std::cout<< b[10] << std::endl;
}

int eraseOverlapIntervals(std::vector<std::vector<int>>& intervals) {
  if(intervals.size() < 2)
    return 0; // 不需要移除  
  //按照终点
  std::sort(intervals.begin(), intervals.end(), [](std::vector<int> a, std::vector<int> b){
    return a[1] < b[1];
  });
  int result = 1;
  int end = intervals[0][1];
  for(int i = 1; i < intervals.size(); i++) {
      if(intervals[i][0] >= end) {
          result++;
          end = intervals[i][1];
      }
  }
  return intervals.size() - result;
}

std::vector<int> twoSum_1(std::vector<int>& numbers, int target) {
  std::vector<int> result;
  int start_index = 0;
  int end_index = numbers.size() - 1;
  while(start_index < end_index) {
      int sum = numbers[start_index] + numbers[end_index];
      if(sum == target)
        return {start_index+1, end_index+1};
      else if(sum < target)
        start_index++;
      else
        end_index--;  
  }
  return result;
}

void merge(std::vector<int>& nums1, int m, std::vector<int>& nums2, int n) {
  int i = 0;
  int j = 0;
  int k = 0;
  std::vector<int> c(m+n, 0);
  while(i < m && j < n) {
      if(nums1[i] <= nums2[j]) {
        c[k++] = nums1[i];
        i++;
      } else {
        c[k++] = nums2[j];
        j++;  
      }
  }
  while(i < m) {
      c[k++] = nums1[i++];
  }
  while(j < n) {
      c[k++] = nums2[j++];
  }
  nums1 = c;        
}

std::string minWindow(std::string s, std::string t) {
  //用长度128的数组映射字符
  std::vector<int> chars(128, 0); //t中字符出现的次数
  std::vector<bool> flag(128, false);//t中是否出现某个字符
  return "hello";
}

int sqrt(int x) {
  if(x == 1)
    return 1;

  int start_value = 0;
  int end_value = x;
  
  while(start_value < end_value) {
      if((end_value - start_value) == 1)
        return start_value;
      int middle_value = start_value + (end_value - start_value) / 2; 
      int product = x  / middle_value;
      if(product == middle_value)
        return middle_value;
      else if(product > middle_value ) {
        start_value = middle_value;
      } else {
        end_value = middle_value;
      }  
  }
  return 0;
}

std::vector<int> searchRange(std::vector<int>& nums, int target) {
  if(nums.size() == 1) {
      if(nums[0] == target)
        return {0, 0};
      else
        return {-1, -1};  
  }
    return {-1,-1};
  int start_index = 0;
  int end_index = nums.size() - 1;
  int target_index = -1;
  while(start_index < end_index) {
      if((end_index - start_index) == 1) {
          if(nums[start_index] != target)
            start_index = end_index;
      }
      int middle_index = start_index + (end_index - start_index) / 2;
      if(nums[middle_index] == target) {
          target_index = middle_index;
          break;
      } else if(nums[middle_index] < target) {
          start_index = middle_index;
      } else {
          end_index = middle_index;
      }
  }
  if(target_index == -1)
    return {-1, -1};
  int l = target_index-1;
  int r = target_index+1;
  while(l >= 0) {
      if(nums[l] == target)
        l--;
      else {
          l++;
          break;
      }  
  }
  l = std::max(l, 0);
  while(r < nums.size()) {
      if(nums[r] == target)
        r++;
      else {
        r--;
        break;  
      }  
  }
  r = std::min(r, static_cast<int>(nums.size()) - 1);
  return {l, r};  
}

bool search1(std::vector<int>& nums, int target) {
  for(int i = 0; i < nums.size(); i++) {
      if(nums[i] == target)
        return true;
  }
  return false;
}

//对l到r之间的数据进行一次快速排序
int quick_selection(std::vector<int>& nums, int l, int r) {
  int i = l+1;
  int j = r;
  int pivot = nums[l];
  while(true) {
      //i小于右边界
      while(i < r && nums[i] <= pivot)
        i++;
      //j大于左边界  
      while(j > l && nums[j] >= pivot)
        j--;
      if(i >= j)
        break;  
      //相遇
      std::swap(nums[i], nums[j]);    
  }
  std::swap(nums[l], nums[j]);
  return j;
}

int findKthLargest(std::vector<int>& nums, int k) {
  int i = 0;
  int j = nums.size()-1;
  int target = nums.size() - k;
  while(i < j) {
      int mid = quick_selection(nums, i, j);
      if(mid == target)
        return nums[mid];
      else if(mid < target) {
          i = mid + 1;
      } else {
          j = mid - 1;
      }  
  }
  return nums[i];
}

void backtracking(int n, int k, int start_index, std::vector<int> result, std::vector<std::vector<int>>& results) {
    if(result.size() == k) {
        results.push_back(result);
        return;
    }
    for(int i = start_index; i <= n; i++) {
        result.push_back(i);
        backtracking(n, k, i+1, result, results);
        result.pop_back();
    }
}

//回溯的减枝优化版本
void backtracking_jianzhi(int n, int k, int start_index, std::vector<int> result, std::vector<std::vector<int>>& results) {
    if(result.size() == k) {
        results.push_back(result);
        return;
    }
    for(int i = start_index; i <= (n - k + 1 + result.size()); i++) {
        result.push_back(i);
        backtracking_jianzhi(n, k, i+1, result, results);
        result.pop_back();
    }
}

//从n个数中选择k个数字的组合问题，利用回溯算法解决
std::vector<std::vector<int>> combine1(int n, int k) {
  std::vector<int> result;
  std::vector<std::vector<int>> results;
  backtracking_jianzhi(n, k, 1, result, results);
  return results;
}

void backtracing_permute(std::vector<int>& nums,
                         std::vector<int> select_indexs,
                         std::vector<int> result, 
                         std::vector<std::vector<int>>& results) {
  if(result.size() == nums.size()) {
    results.push_back(result);
    return;
  }
  for(int i = 0; i < nums.size(); i++) {
    auto iter = std::find(select_indexs.begin(), select_indexs.end(), i);
    if(iter != select_indexs.end())
      continue;
    else
      select_indexs.push_back(i);
    result.push_back(nums[i]);
    backtracing_permute(nums, select_indexs, result, results);
    select_indexs.pop_back();
    result.pop_back();  
  }
}

//全排列
std::vector<std::vector<int>> permute(std::vector<int>& nums) {
  std::vector<int> result;
  std::vector<std::vector<int>> results;
  std::vector<int> select;
  backtracing_permute(nums,select, result, results);
  return results;
}

void backtracing(int start_index,
          std::unordered_map<char, std::vector<char>>& char_map, 
          const std::string& digits, 
          std::string result, 
          std::vector<std::string>& results) {
  if(result.size() == digits.size()) {
    results.push_back(result);
    return;
  }
  if(start_index >= digits.size())
    return;
  std::vector<char> digit_char = char_map[digits[start_index]];
  for(int j = 0; j < digit_char.size(); j++) {
    result.push_back(digit_char[j]);
    backtracing(start_index + 1,char_map, digits, result, results);
    result.pop_back();
  }
}

std::vector<std::string> letterCombinations(std::string digits) {
  if(digits.empty())
    return {};
  std::unordered_map<char, std::vector<char>> char_map;
  char_map['2'] = {'a', 'b', 'c'};
  char_map['3'] = {'d', 'e', 'f'};
  char_map['4'] = {'g', 'h', 'i'};
  char_map['5'] = {'j', 'k', 'l'};
  char_map['6'] = {'m','n', 'o'};
  char_map['7'] = {'p', 'q', 'r', 's'};
  char_map['8'] = {'t', 'u', 'v'};
  char_map['9'] = {'w', 'x', 'y','z'};
  std::string result;
  std::vector<std::string> results;
  backtracing(0, char_map, digits, result, results);
  return results; 
}

// 并查集
class UF {
  public:
  UF(int n) {
    father = std::vector<int>(n);
    //初始的时候，每个人的代理人是他自己
    std::iota(father.begin(), father.end(), 0);
  }

  //循环的对代理人进行追溯
  int getFather(int p) {
    while(p != father[p]) {
      p = father[p];
    }
    return p;
  }

  //链接两个节点,f[3的代表人] = 4的代表人
  void connect(int p, int q) {
    father[getFather(p)] = getFather(q); 
  }
  
  bool isConnected(int p, int q) {
    return getFather(p) == getFather(q);
  }


  std::vector<int> father;
};

std::vector<int> findRedundantConnection(std::vector<std::vector<int>>& edges) {
  int N = edges.size();
  UF uf(N+1);
  for(int i = 0; i < edges.size(); i++) {
    auto edge = edges[i];
    if(uf.isConnected(edge[0], edge[1]))
      return edge;
    uf.connect(edge[0], edge[1]);  
  }
  return {-1, -1};
}

class LRUCache {
public:
    LRUCache(int capacity) {
      capacity_ = capacity;
    }
    
    int get(int key) {
      if(elements_.count(key) != 0) {
        for(auto iter = ages_.begin(); iter != ages_.end(); iter++) {
          if((*iter) == key) {
            ages_.erase(iter);
            ages_.push_front(key);
            break;
          }
        }
        return elements_[key]; 
      }
      return -1;  
    }
    
    void put(int key, int value) {
      if(elements_.count(key) != 0) {
        elements_[key] = value;
        for(auto iter = ages_.begin(); iter != ages_.end(); iter++) {
          if((*iter) == key) {
            ages_.erase(iter);
            ages_.push_front(key);
            break;
          }
        }
      } else {
        if(elements_.size() < capacity_) {
          elements_[key] = value;
        }
        else {
          int oldest_key = ages_.back();
          ages_.pop_back();
          elements_.erase(oldest_key);
          elements_[key] = value;
        }
        ages_.push_front(key);  
      }
    }
    
    std::unordered_map<int, int> elements_;
    //从最新的到最老的
    std::list<int> ages_;
    int capacity_;
};

int dfs_islands(int i, int j, 
               std::vector<std::vector<int>>& grid,
               std::vector<std::vector<bool>>& visited) {
  if(i < 0 || i >= grid.size() || j < 0 || j >= grid[0].size())
    return 0;               
  if(visited[i][j])
    return 0;
  visited[i][j] = true;
  if(grid[i][j] == 0)
    return 0;
  else
    return 1 + dfs_islands(i-1,j,grid,visited) +
    dfs_islands(i+1, j, grid, visited) + 
    dfs_islands(i, j-1, grid, visited) +
    dfs_islands(i, j+1, grid, visited);
}

int maxAreaOfIsland(std::vector<std::vector<int>>& grid) {
  int rows = grid.size();
  int cols = grid[0].size();
  std::vector<std::vector<bool>> visited(rows, std::vector<bool>(cols, false));
  int max_area = 0;
  for(int i = 0; i < rows; i++) {
    for(int j = 0; j < cols; j++) {
      max_area = std::max(max_area, dfs_islands(i, j, grid, visited));
    }
  }
  return max_area;
}

//对于第k个人,遍历他所有的朋友
void dfs_friends(int k,
               std::vector<std::vector<int>>& M,
               std::vector<bool>& visited) {
  if(k < 0 || k >= M.size())
    return;               
  if(visited[k])
    return;
  visited[k] = true;
  for(int i = 0; i < M.size(); i++) {
    if(M[k][i] == 1 && !visited[i])
      dfs_friends(i, M, visited);
  }
}

/*
1 0 0 1
0 1 1 0
0 1 1 1
1 0 1 1
*/
int findCircleNum(std::vector<std::vector<int>>& M) {
  int result = 0;
  int rows = M.size();
  int cols = M[0].size();
  std::vector<bool> visited(rows, false);
  for(int i = 0; i < rows; i++) {
    if(!visited[i]) {
      dfs_friends(i, M, visited);
      result++;
    }
  }
  return result;
}

//对图进行遍历，广度优先搜索进行着色
bool isBipartite1(std::vector<std::vector<int>>& graph) {
  int n = graph.size();
  std::vector<int> color(n, 0);
  std::queue<int> q;
  q.push(0);
  color[0] = 1;
  while(!q.empty()) {
    int node = q.front();
    q.pop();
    std::vector<int> connected_nodes = graph[node];
    for(int i = 0; i < connected_nodes.size(); i++) {
      if(color[connected_nodes[i]] == 0) {
        color[connected_nodes[i]] = color[node] == 1 ? 2 : 1;
        q.push(connected_nodes[i]);
      } else {
        if(color[connected_nodes[i]] == color[node])
          return false;
      }
    }
  }
  return true;
}

bool isBipartite(std::vector<std::vector<int>>& graph) {
  int n = graph.size();
  std::vector<int> color(n, 0);
  std::queue<int> q;
  for(int i = 0; i < n; i++) {
    if(color[i] == 0) {
      q.push(i);
      color[i] = 1;
    }

    while(!q.empty()) {
      int node = q.front();
      q.pop();
      std::vector<int> connected_nodes = graph[node];
      for(int i = 0; i < connected_nodes.size(); i++) {
        if(color[connected_nodes[i]] == 0) {
          color[connected_nodes[i]] = color[node] == 1 ? 2 : 1;
          q.push(connected_nodes[i]);
        } else {
          if(color[connected_nodes[i]] == color[node])
            return false;
        }
      }
    }
  }
  return true;
}

int uniquePaths(int m, int n) {
  std::vector<std::vector<int>> paths(m, std::vector<int>(n, 1));
  for(int i = 1; i < m; i++) {
    for(int j = 1; j < n; j++) {
      paths[i][j] = paths[i-1][j] + paths[i][j-1];
    }
  }
  return paths[m-1][n-1];
}

//拓扑排序问题，构建领接矩阵
std::vector<int> findOrder(int numCourses, std::vector<std::vector<int>>& prerequisites) {
  std::vector<std::vector<int>> graph(numCourses, std::vector<int>());
  std::vector<int> indegree(numCourses, 0);
  for(int i = 0; i < prerequisites.size();i++) {
    //依赖于[1]的所有元素
    graph[prerequisites[i][1]].push_back(prerequisites[i][0]);
    indegree[prerequisites[i][0]]++;
  }
  std::vector<int> result;
  //利用队列做广度优先遍历
  std::queue<int> q;
  for(int i = 0; i < numCourses; i++) {
    if(!indegree[i])
      q.push(i);
  }
  while(!q.empty()) {
    int v = q.front();
    result.push_back(v);
    q.pop();
    for(auto& e : graph[v]) {
      indegree[e]--;
      if(indegree[e] == 0)
        q.push(e);
    }
  }
  for(int i = 0; i < numCourses; i++) {
    if(indegree[i] != 0)
      return {};
  }
  return result;
}

int diameterOfBinaryTree(TreeNode* root) {
  if(root == nullptr)
    return 0;
  int length_root = maxDepth(root->left_child) +  maxDepth(root->right_child);
  int length_left = diameterOfBinaryTree(root->left_child);
  int length_right = diameterOfBinaryTree(root->right_child);
  return std::max(std::max(length_left, length_right), length_root);
}

int pathSum1(TreeNode* root, int sum) {
    int result = 0;
    if(root == nullptr)
      return 0;
    if(root->val == sum)
      result = 1;
    if(root->left_child == nullptr && root->right_child == nullptr)
      return result;   
    result = result + pathSum1(root->left_child, sum - root->val) +
            pathSum1(root->right_child, sum - root->val) +
            pathSum1(root->left_child, sum) +
            pathSum1(root->right_child, sum);
    return result;        
}

//把二进制逆序
uint32_t reverseBits(uint32_t n) {
  int result = 0;
  //先左移一位，第一位是符号位
  for(int i = 0; i < 32; i++) {
    result <<= 1;
    result += (n & 1);
    n >>=1;
  }
  return result;     
}

bool isPowerOfFour(int n) {
  if(n == 1)
    return true;
  if(n < 4)
    return false;  
  if(n % 4 != 0)
    return false;
  return isPowerOfFour(n / 4);  
}

bool isContainSameChar(const std::string& a,
                       const std::string& b) {
  std::unordered_map<char, bool> map_a;
  for(int i =0; i < a.size(); i++) {
    map_a[a[i]] = true;
  }
  for(int i = 0; i < b.size(); i++) {
    if(map_a.count(b[i]))
      return true;
  }
  return false;
}

int maxProduct1(std::vector<std::string>& words) {
  int result = 0;
  for(int i = 0; i < words.size(); i++) {
    for(int j = i+1; j < words.size(); j++) {
      if(!isContainSameChar(words[i], words[j]))
        result = std::max(result, int(words[i].size()) * int(words[j].size()));
    }
  }
  return result;
}

int maxProduct(std::vector<std::string>& words) {
  std::vector<int> int_words;
  //通过位运算检查单词是否相同
  for(auto& word : words) {
    int value = 0;
    for(auto& c : word)
      value |= (1 << (c - 'a'));
    int_words.push_back(value);  
  }
  int result = 0;
  for(int i = 0; i < words.size(); i++) {
    for(int j = i+1; j < words.size(); j++) {
      if(!(int_words[i] & int_words[j]))
        result = std::max(result, int(words[i].size()) * int(words[j].size()));
    }
  }
  return result;
}

//利用动态规划求解
std::vector<int> countBits(int num) {
  std::vector<int> results(num+1, 0);
  for(int i = 1; i <= num; i++) {
    if((i & 1) == 0) {
      results[i] = results[i >> 1];
    } else {
      results[i] = results[i-1] + 1;
    }
  }
  return results;
}

bool containsDuplicate(std::vector<int>& nums) {
  std::unordered_map<int, int> num_count;
  for(int i = 0; i < nums.size(); i++) {
    if(num_count.count(nums[i]) == 0)
      num_count[nums[i]] = 1;
    else
      return true;  
  }
  return false;
}

int countPrimes(int n) {
  if(n <= 2)
    return 0;
  std::vector<bool> prime(n, true);
  int count = n-2;
  for(int i = 2; i < n; i++) {
    for(int j = 2 * i; j < n; j+=i) {
      if(prime[j]) {
        prime[j] = false;
        count--;
      }
    }
  }
  return count;  
}

std::string convertToBase7(int num) {
  if(num == 0)
    return "0";
  bool is_negative = num < 0;
  if(is_negative)
    num = -num;
  std::string result="";  
  while(num) {
    result = std::to_string(num % 7) + result;
    num /= 7;
  }
  if(is_negative)
    return "-" + result;
  return result;    
}

int trailingZeroes(int n) {
  return n == 0? 0: n / 5 + trailingZeroes(n / 5);
}

bool isPowerOfThree(int n) {
  if(n == 1)
    return true;
  if(n < 3)
    return false;  
  if(n % 3 != 0)
    return false;
  return isPowerOfFour(n / 3);
}

class Solution1 {
public:
    /** @param head The linked list's head.
        Note that the head is guaranteed to be not null, so it contains at least one node. */
    Solution1(ListNode* head) {
      ListNode* p = head;
      while(p) {
        nums.push_back(p->val);
        p = p->next;
      }
    }
    
    /** Returns a random node's value. */
    int getRandom() {
      int random_index = random() % nums.size();
      return nums[random_index];
    }

    std::vector<int> nums;
};

class Solution2 {
public:
Solution2(std::vector<int>& nums) {
  origin_nums = nums;
}

/** Resets the array to its original configuration and return it. */
std::vector<int> reset() {
  return origin_nums;
}

/** Returns a random shuffling of the array. */
std::vector<int> shuffle() {
  // fisher-yate洗牌算法
  if(origin_nums.empty())
    return {};
  std::vector<int> shuffle(origin_nums);
  int n = shuffle.size();
  for(int i = 0; i < n; i++) {
    int pos = random() % (n-i); //随机区间被逐步的缩小
    std::swap(shuffle[i], shuffle[i+pos]);
  }
  return shuffle;
}

std::vector<int> origin_nums;
};

//按照权重来进行随机的选择
//使用前缀和来模拟
class Solution3 {
public:
    Solution3(std::vector<int>& w) {
      nums = w;
    }
    
    int pickIndex() {

    }

    std::vector<int> nums;
};

char findTheDifference(std::string s, std::string t) {
  std::unordered_map<char, int> char_map;
  for(int i = 0; i < s.size(); i++) {
    if(char_map.count(s[i]) == 0)
      char_map[s[i]] = 1;
    else
      char_map[s[i]]++;  
  }
  for(int i = 0; i < t.size(); i++) {
    char_map[t[i]]--;
  }
  for(auto iter = char_map.begin(); iter != char_map.end(); iter++) {
    if(iter->second != 0)
      return iter->first;
  }
  return s.front();
}

int numberOfArithmeticSlices(std::vector<int>& A) {
    int n = A.size();
    if(n < 3)
      return 0;
    std::vector<int> dp(n, 0);
    for(int i = 2; i < n; i++) {
        if((A[i] - A[i-1]) == (A[i-1] - A[i-2]))
          dp[i] = dp[i-1] + 1;
    }
    return std::accumulate(dp.begin(), dp.end(), 0);
}

bool isIsomorphic(std::string s, std::string t) {
    if(s.size() != t.size())
      return false;
    bool result = true;
    std::map<char, char> map_s_t;
    for(int i = 0; i < s.size(); i++) {
        if(map_s_t.count(s[i]) != 0) {
          if(map_s_t[s[i]] != t[i])
            return false;
        } else {
          map_s_t[s[i]] = t[i];
        }
    }

    std::map<char, char> map_t_s;
        for(int i = 0; i < t.size(); i++) {
        if(map_t_s.count(t[i]) != 0) {
          if(map_t_s[t[i]] != s[i])
            return false;
        } else {
          map_t_s[t[i]] = s[i];
        }
    }
    return result;
}

// 01矩阵问题，　method1: 广度优先搜索
std::vector<std::vector<int>> updateMatrix1(std::vector<std::vector<int>>& matrix) {
    int dir[4][2] = {{-1, 0}, {1, 0},{0, 1}, {0, -1}};
    int m = matrix.size();
    int n = matrix.front().size();
    std::vector<std::vector<int>> dist(m, std::vector<int>(n, 0));
    std::vector<std::vector<int>> visited(m, std::vector<int>(n, false));
    std::queue<std::pair<int, int>> q;
    for(int i= 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            if(matrix[i][j] == 0) {
              q.push({i, j});
              visited[i][j] = true;  
            }
        }
    }
    while(!q.empty()) {
        auto pos = q.front();
        q.pop();
        for(int i = 0; i < 4; i++) {
            int new_i = pos.first + dir[i][0];
            int new_j = pos.second + dir[i][1];
            if(new_i >=0 && new_i < m && new_j >= 0 && new_j < n && !visited[new_i][new_j]) {
                visited[new_i][new_j] = true;
                dist[new_i][new_j] = dist[pos.first][pos.second]+1;
                q.push({new_i, new_j});
            }
        }

    }
    return dist;
}

// 01矩阵问题，　method1: 动态规划　two pass 每个位置的dist是四种移动方法的动态规划，参照leetcode题解
std::vector<std::vector<int>> updateMatrix(std::vector<std::vector<int>>& matrix) {
    int m = matrix.size();
    int n = matrix.front().size();
    //初始的时候将每个位置的距离设为一个很大的值
    std::vector<std::vector<int>> dist(m, std::vector<int>(n, INT32_MAX / 2));
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            if(matrix[i][j] == 0)
              dist[i][j] = 0;
        }
    }
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            if((i - 1) >= 0)
              dist[i][j] = std::min(dist[i][j], dist[i-1][j] + 1);
            if((j - 1) >= 0)
              dist[i][j] = std::min(dist[i][j], dist[i][j-1] + 1); 
        }
    }
    for(int i = m -1; i >= 0; i--) {
        for(int j = n - 1; j >= 0; j--){
            if((i+1) < m)
              dist[i][j] = std::min(dist[i][j], dist[i+1][j] + 1);
            if((j+1) < n)
              dist[i][j] = std::min(dist[i][j], dist[i][j+1] + 1);  
        }
    }
    return dist;
}

int maximalSquare(std::vector<std::vector<char>>& matrix) {
  int m = matrix.size();
  int n = matrix.front().size();
  // dp[i][j]表示以i,j为右下角的正方形的边长
  std::vector<std::vector<int>> dp(m, std::vector<int>(n, 0));
  int max_size = 0;
  for(int i = 0; i < m; i++) {
      for(int j = 0; j < n; j++) {
          if(matrix[i][j] == '1') {
              if(i == 0 || j == 0)
                dp[i][j] = 1;
              else {
                dp[i][j] = std::min(dp[i-1][j-1], std::min(dp[i-1][j], dp[i][j-1])) + 1;
              }
              max_size = std::max(max_size, dp[i][j]);
          }
      }
  }
  return max_size * max_size;
}

// 1 4 9 6 25
int numSquares(int n) {
    std::vector<int> dp(n+1, std::numeric_limits<int>::max());
    dp[0] = 0;
    for(int i = 1; i <=n; i++) {
        for(int j = 1; j * j < i; j++) {
            dp[i] = std::min(dp[i], dp[i - j * j] + 1);
        }
    }
    return dp[n];
}

std::vector<int> findDisappearedNumbers(std::vector<int>& nums) {
    int n = nums.size();
    std::vector<int> contained(n+1, -1);
    for(int i = 0; i < nums.size(); i++) {
        contained[nums[i]] = 1;
    }
    std::vector<int> result;
    for(int i = 1; i <=n; i++) {
        if(contained[i] == -1)
          result.push_back(i);
    }
    return result;
}

//也可以先水平翻转在转置
void rotate(std::vector<std::vector<int>>& matrix) {
    int m = matrix.size();
    std::vector<std::vector<int>> result(m, std::vector<int>(m));
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < m; j++) {
            result[j][m-i-1] = matrix[i][j];
        }
    }
    matrix = result;
}

bool searchMatrix(std::vector<std::vector<int>>& matrix, int target) {
    int m = matrix.size();
    int n = matrix.front().size();
    int i = 0;
    int j = n-1;
    while(1) {
        if(matrix[i][j] == target)
          return true;
        else if(matrix[i][j] > target)
          j--;
        else
          i++;
        if(j < 0 || i >= m)
          return false;      
    }
    return true;    
}


// 1 0 2 3 4
//到当前为止的最大值＝索引的时候可以完成一次分割
int maxChunksToSorted(std::vector<int>& arr) {
    int n = arr.size();
    int cur_max = 0;
    int chuck = 0;
    for(int i = 0; i < n; i++) {
        cur_max = std::max(arr[i], cur_max);
        if(cur_max == i)
          chuck++;
    }
    return chuck;
}

class MinStack {
public:
    /** initialize your data structure here. */
    MinStack() {
        
    }
    
    void push(int x) {
        norm_stack.push(x);
        if(min_stack.empty())
          min_stack.push(x);
        else {
          int top = min_stack.top();
          min_stack.push(std::min(top, x));
        }
    }
    
    void pop() {
        norm_stack.pop();
        min_stack.pop();
    }
    
    int top() {
        return norm_stack.top();
    }
    
    int getMin() {
        return min_stack.top();
    }

    std::stack<int> norm_stack;
    std::stack<int> min_stack;
};

bool wordBreak(std::string s, std::vector<std::string>& wordDict) {
    // dp[i] 表示以第i个字母结尾是否满足条件
    int n = s.size();
    std::vector<bool> dp(n+1, false);
    dp[0] = true;
    for(int i = 1; i<= n; i++) {
        for(auto& word : wordDict) {
            int len = word.length();
            if(i >= len && s.substr(i - len, len) == word)
              dp[i] = dp[i] | dp[i-len];
        }
    }
    return dp[n];
}

int lastStoneWeight(std::vector<int>& stones) {
    std::sort(stones.begin(), stones.end());
    while(stones.size() > 2) {
      int tmp_size = stones[stones.size() - 1] - stones[stones.size() - 2];
      if(tmp_size == 0) {
        stones.erase(stones.end() - 2, stones.end());
      } else {
        stones.erase(stones.end() - 2, stones.end());
        for(int i = 0; i < stones.size(); i++) {
          if(tmp_size <= stones[i]) {
            stones.insert(stones.begin() + i, tmp_size);
            break;
          }
        }
        if(tmp_size > stones.back())
          stones.push_back(tmp_size);
      }
    }
    if(stones.size() == 1)
      return stones[0];
    return stones[1] - stones[0];  
}

bool canPlaceFlowers(std::vector<int>& flowerbed, int n) {
  int res = n;
  for(int i = 0; i < flowerbed.size(); i++) {
    if(flowerbed[i] == 1)
      continue;
    if(i >= 1 && flowerbed[i-1] == 1)
      continue;
    if(i < flowerbed.size() -1 && flowerbed[i+1] == 1)
      continue;
    flowerbed[i] = 1;
    res--;
    if(res == 0)
      return true;
  }
  return res <= 0;
}

bool canWinNim(int n) {
  return n % 4 != 0;
}

//记录一个数字左边的乘积和右边的乘积
//左边的乘积和右边的乘积
std::vector<int> productExceptSelf(std::vector<int>& nums) {
  std::vector<int> left_product(nums.size(), 0);
  std::vector<int> right_product(nums.size(), 0);
  int tmp_left_product = 1;
  int tmp_right_product = 1;
  for(int i = 0; i < nums.size(); i++) {
    left_product[i] = tmp_left_product;
    tmp_left_product *= nums[i];
  }
  for(int i = nums.size() - 1; i >=0; i--) {
    right_product[i] = tmp_right_product;
    tmp_right_product *= nums[i];
  }
  std::vector<int> result(nums.size(), 0);
  for(int i = 0; i < nums.size(); i++) {
    result[i] = left_product[i] * right_product[i];
  }
  return result;
}

//改变当前节点的值，让当前节点指向下下个节点
void deleteNode(ListNode* node) {
    node->val = node->next->val;
    node->next = node->next->next;
}

std::string reverseWords(std::string s) {
  std::vector<std::string> words;
  std::string tmp_word;
  for(int i = 0; i < s.size(); i++) {
    if(s[i] == ' ' && !tmp_word.empty()) {
      words.push_back(tmp_word);
      tmp_word.clear();
    } else
      tmp_word.push_back(s[i]);
  }
  if(!tmp_word.empty())
    words.push_back(tmp_word);
  std::string result = "";
  for(int i = 0; i < words.size(); i++) {
    std::string word = words[i];
    std::cout<< word << std::endl;
    std::reverse(word.begin(), word.end());
    result += word;
    if(i != (words.size() - 1))
      result += " ";
  }
  return result;
}

ListNode* sortList(ListNode* head) {
  std::vector<int> nums;
  ListNode* p = head;
  while(p) {
    nums.push_back(p->val);
    p = p->next;
  }
  std::sort(nums.begin(), nums.end());
  ListNode* q = head;
  int start_index = 0;
  while(q) {
    q->val = nums[start_index++];
    q = q->next;
  }
  return head;
}

//两两合并
ListNode* mergeKLists(std::vector<ListNode*>& lists) {
    if(lists.empty())
      return nullptr;
    if(lists.size() < 2)
      return lists.front();
    ListNode* k = mergeTwoLists(lists[0], lists[1]);
    for(int i = 2; i < lists.size(); i++) {
      k = mergeTwoLists(k, lists[i]);
    }
    return k;
}

bool isUnique(std::string astr) {
  std::unordered_map<char, int> char_count;
  for(auto c : astr) {
    if(char_count.count(c) == 0)
      char_count[c] = 1;
    else
    {
      return false;
    }   
  }
  return true;
}

bool CheckPermutation(std::string s1, std::string s2) {
  if(s1.size() != s2.size())
    return false;
  std::unordered_map<char, int> char_count1;
  std::unordered_map<char, int> char_count2;
  for(int i = 0; i < s1.size(); i++) {
    if(char_count1.count(s1[i]) == 0)
      char_count1[s1[i]] = 1;
    else
      char_count1[s1[i]]++;
    
    if(char_count2.count(s2[i]) == 0)
      char_count2[s2[i]] = 1;
    else
      char_count2[s2[i]]++;
  }

  for(auto iter = char_count1.begin(); iter != char_count1.end(); iter++) {
    auto c = iter->first;
    if(char_count2.count(c) == 0)
      return false;
    if(char_count1[c] != char_count2[c])
      return false;  
  }
  return true;
}

std::string replaceSpaces(std::string S, int length) {
  std::string result;
  for(int i = 0; i < length; i++) {
    if(S[i] != ' ')
      result.push_back(S[i]);
    else {
      result.push_back('/%');
      result.push_back('2');
      result.push_back('0');
    }
  }
  return result;
}

bool canPermutePalindrome(std::string s) {
  std::unordered_map<char, int> char_count;
  for(int i = 0; i < s.size(); i++) {
    if(char_count.count(s[i]) == 0)
      char_count[s[i]] = 1;
    else
      char_count[s[i]]++;
  }
  int odd_count = 0;
  for(auto iter = char_count.begin(); iter != char_count.end(); iter++) {
    auto c = iter->first;
    if((iter->second % 2) != 0)
      odd_count++;
    if(odd_count > 1)
      return false;  
  }
  return true;
}

bool oneEditAway(std::string first, std::string second) {
  if(std::abs(int(first.size()) - int(second.size())) > 1)
    return false;
  int first_size = first.size();
  int second_size = second.size();
  std::cout<< first_size <<" " << second_size << std::endl;
  if(first_size == second_size) {
    int diff_count = 0;
    for(int i = 0; i < first.size(); i++) {
        if(first[i] != second[i])
          diff_count++;
        if(diff_count > 1)
          return false;  
    }
    return true;
  }

  if(first_size > second_size) {
    int diff_index = -1;
    for(int i = 0; i < second.size(); i++) {
        if(first[i] != second[i]) {
          diff_index = i;
          break;
        }
    }
    if(diff_index == -1)
      return true;
    else {
        for(int i = diff_index + 1; i< first.size(); i++) {
            if(first[i] != second[i-1])
              return false;
        }
    }
    return true;  
  }  

  if(first_size < second_size) {
    int diff_index = -1;
    for(int i = 0; i < first.size(); i++) {
        if(first[i] != second[i]) {
          diff_index = i;
          break;
        }
    }
    if(diff_index == -1)
      return true;
    else {
        for(int i = diff_index + 1; i< second.size(); i++) {
            if(second[i] != first[i-1])
              return false;
        }
    }
    return true;
  }
  
  return false;
}

std::string compressString(std::string S) {
  std::vector<std::pair<int, char>> char_count;
  char pre_char = S[0];
  int count = 1;
  for(int i = 1; i < S.size(); i++) {
    if(S[i] == S[i-1])
      count++;
    else {
      char_count.push_back(std::make_pair(count, pre_char));
      count = 1;
      pre_char = S[i];
    }  
  }
  char_count.push_back(std::make_pair(count, pre_char));
  std::string result = "";
  for(int i = 0; i < char_count.size(); i++) {
    result += std::to_string(char_count[i].first);
    result += char_count[i].second;
  }
  if(result.size() < S.size())
    return result;
  return S;  
}

int kthToLast(ListNode* head, int k) {
  ListNode* p = head;
  ListNode* q = p;
  for(int i= 0 ; i<k; i++) {
    q = q->next;
  }
  while(q) {
    p = p->next;
    q = q->next;
  }
  return p->val;
}

void setZeroes(std::vector<std::vector<int>>& matrix) {
  std::vector<std::pair<int, int>> zero_localization;
  for(int i = 0; i < matrix.size(); i++) {
    for(int j = 0; j < matrix[0].size(); j++) {
      if(matrix[i][j] == 0)
        zero_localization.push_back(std::make_pair(i, j));
    }
  }

  for(int i = 0; i < zero_localization.size(); i++) {
    int row = zero_localization[i].first;
    int col = zero_localization[i].second;
    //清除这一行
    for(int tmp_col = 0; tmp_col < matrix[0].size(); tmp_col++) {
      matrix[row][tmp_col] = 0;
    }
    //清除这一列
    for(int tmp_row = 0; tmp_row < matrix.size(); tmp_row++) {
      matrix[tmp_row][col] = 0;
    }
  }
}

//用我做的东西去面试uniform 把路牌加进去更好的完成初始化
bool isFlipedString(std::string s1, std::string s2) {
  if(s1.size() != s2.size())
    return false;
  if(s1.empty())
    return true;  
  s1 = s1 + s1;
  for(int i =0 ; i< s2.size(); i++) {
    std::string tmp_str = s1.substr(i, s2.size());
    if(tmp_str == s2)
      return true;
  }
  return false;  
}

ListNode* removeDuplicateNodes(ListNode* head) {
  if(head == nullptr)
    return nullptr;
  std::unordered_map<int, int> num_count;
  ListNode *p = head;
  num_count[p->val] = 1;
  ListNode* q = p->next;
  while(q) {
    if(num_count.count(q->val) == 0) {
      num_count[q->val] = 1;
      p->next = q;
      p = q;
      q = q->next; 
    } else {
      q = q->next;
    }
  }
  p->next = nullptr;
  return head;
}

class TripleInOne {
public:
    TripleInOne(int stackSize) {
      nums.resize(stackSize);
    }
    
    void push(int stackNum, int value) {
      nums[stackNum].push_back(value);
    }
    
    int pop(int stackNum) {
      if(isEmpty(stackNum))
        return -1;
      int result = nums[stackNum].back();
      nums[stackNum].erase(nums[stackNum].end()-1);
      return result;
    }
    
    int peek(int stackNum) {
      return nums[stackNum].back();
    }
    
    bool isEmpty(int stackNum) {
      return nums[stackNum].empty();
    }

    std::vector<std::vector<int>> nums;
};

bool isPalindrome(ListNode* head) {
  ListNode* p = head;
  ListNode* q = head;
  std::vector<int> nums;
  while(p) {
    nums.push_back(p->val);
    p = p->next;
  }
  for(int i = 0; i < nums.size() / 2; i++) {
    if(nums[i] != nums[nums.size() -1 -i])
      return false;    
  }
  return true;
}

ListNode *getIntersectionNode1(ListNode *headA, ListNode *headB) {
  if(headA == nullptr || headB == nullptr)
    return nullptr;
  ListNode* p = headA;
  ListNode* q = headB;
  int count = 1;
  while(1) {
    if(p == q)
      return p;    
    p = p->next;
    q = q->next;
    if(p == nullptr) {
      p = headB;
      count++;
      if(count > 1)
        return nullptr;
    }
    if(q == nullptr)
      q = headA;
  }
  return nullptr;
}

ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
  ListNode* result = new ListNode(0);
  ListNode* p = result;
  int jinwei = 0;
  while(l1 && l2) {
    int num = l1->val + l2->val + jinwei;
    jinwei = num / 10;
    num %= 10;
    ListNode* node = new ListNode(num);
    p->next = node;
    p = node;
    l1 = l1->next;
    l2 = l2->next;
  }
  while(l1) {
    int num = l1->val + jinwei;
    jinwei = num / 10;
    num %= 10;
    ListNode* node = new ListNode(num);
    p->next = node;
    p = node;
    l1 = l1->next;
  }
  while(l2) {
    int num = l2->val + jinwei;
    jinwei = num / 10;
    num %= 10;
    ListNode* node = new ListNode(num);
    p->next = node;
    p = node;
    l2 = l2->next;
  }
  if(jinwei)
    p->next = new ListNode(1);
  return result->next;  
}

//相遇的时候再分别移动一步就可以再次相遇
ListNode *detectCycle1(ListNode *head) {
            if(head == nullptr)
      return nullptr;
    ListNode* fast = head;
    ListNode* slow = head;
    while(1) {
      slow = slow->next;
      if(slow == nullptr)
        return nullptr;
      if(fast == nullptr)
        return nullptr;  
      fast = fast->next;
      if(fast == nullptr)
        return nullptr;
      fast = fast->next;
      if(fast == slow)
        break;    
    }
    fast = head;
    while(1) {
      if(fast == slow)
        return fast;
      fast = fast->next;
      slow = slow->next;  
    }
    return nullptr;
}

class AnimalShelf {
public:
    AnimalShelf() {

    }
    
    void enqueue(std::vector<int> animal) {
      if(animal[1] == 0) {
        cat_q.push_back(animal);
      }
      else
        dog_q.push_back(animal);
      all.push_back(animal);      
    }
    
    std::vector<int> dequeueAny() {
      if(all.empty())
        return {-1, -1};
      if(all[0][1] == 0)
        cat_q.erase(cat_q.begin());
      else
        dog_q.erase(dog_q.begin());
      std::vector<int> result = all.front();
      all.erase(all.begin());
      return result;      
    }
    
    std::vector<int> dequeueDog() {
      if(dog_q.empty())
        return {-1, -1};
      std::vector<int> result = dog_q.front();
      int id = result[0];
      for(int i = 0; i < all.size(); i++) {
        if(all[i][0] == id)
          all.erase(all.begin() + i);
      }
      dog_q.erase(dog_q.begin());
    }
    
    std::vector<int> dequeueCat() {
      if(cat_q.empty())
        return {-1, -1};
      std::vector<int> result = cat_q.front();
      int id = result[0];
      for(int i = 0; i < all.size(); i++) {
        if(all[i][0] == id)
          all.erase(all.begin() + i);
      }
      cat_q.erase(cat_q.begin());
    }

    //存储了狗的编号
    std::vector<std::vector<int>> dog_q;
    std::vector<std::vector<int>> cat_q;

    std::vector<std::vector<int>> all;
};

bool findExistPathHelper(int n, std::vector<std::vector<int>>& connections, int start, int target, std::vector<std::vector<bool>>& visited) {
  if(connections[start][target] == 1)
    return true;
  for(int i = 0; i < n; i++) {
    if(start == i)
      continue;
    if(visited[start][i])
      continue;
    visited[start][i] = true;
    if(connections[start][i] == 1) {
      return findExistPathHelper(n, connections, i, target, visited);
    }  
  }  
}

bool findWhetherExistsPath(int n, std::vector<std::vector<int>>& graph, int start, int target) {
  std::vector<std::vector<int>> connections(n, std::vector<int>(n, 0));
  for(int i = 0; i < graph.size(); i++) {
    auto connection = graph[i];
    connections[connection[0]][connection[1]] = 1;
  }
  std::vector<std::vector<bool>> visited(n, std::vector<bool>(n, false));
  return findExistPathHelper(n, connections, start, target, visited);
}

TreeNode* sortedArrayToBST(std::vector<int>& nums) {
   if(nums.size() == 0)
     return nullptr;
   if(nums.size() == 1)
     return new TreeNode(nums[0]);

   int mid = nums[nums.size() / 2];
   TreeNode* result = new TreeNode(mid);
   std::vector<int> left(nums.begin(), nums.begin() + nums.size() / 2);
   std::vector<int> right(nums.begin() + nums.size() / 2 + 1, nums.end());
   result->left_child = sortedArrayToBST(left);
   result->right_child = sortedArrayToBST(right);
   return result;
}

std::vector<ListNode*> listOfDepth(TreeNode* tree) {
  std::vector<ListNode*> result;
  if(tree == nullptr)
    return result;
  std::queue<TreeNode*> q_tree;
  q_tree.push(tree);
  result.push_back(new ListNode(tree->val));
  while(!q_tree.empty()) {
    std::vector<TreeNode*> cur_level;
    ListNode* tmp = new ListNode(-1);
    ListNode* p = tmp;
    while(!q_tree.empty()) {
      TreeNode* node = q_tree.front();
      q_tree.pop();
      if(node->left_child)
        cur_level.push_back(node->left_child);
      if(node->right_child)
        cur_level.push_back(node->right_child);
    }
    for(int i = 0; i < cur_level.size(); i++) {
      q_tree.push(cur_level[i]);
      p->next = new ListNode(cur_level[i]->val);
      p = p->next;
    }
    if(tmp->next)
      result.push_back(tmp->next);
  }
  return result;
}

int waysToStep(int n) {
  if(n == 1)
    return 1;
  if(n == 2)
    return 2;
  if(n == 3)
    return 4;    
  std::vector<long> results(n+1 ,1);
  results[1] = 1;
  results[2] = 2;
  results[3] = 4;
  for(int i = 4; i <= n; i++) {
    results[i] = (results[i-3] % 1000000007 + results[i-2] % 1000000007 + results[i-1] %1000000007) % 1000000007;
  }
  return results[n];
}

std::vector<std::vector<int>> pathWithObstacles(std::vector<std::vector<int>>& obstacleGrid) {
  
}

int main()
{
  TripleInOne a(1);
  a.push(0, 1);
  a.push(0, 2);
  std::cout<< a.pop(0) << std::endl;
  //std::cout<< a.pop(0) << std::endl;
  //std::cout<< a.pop(0) << std::endl;
  //std::cout<< a.isEmpty(0) << std::endl;
  return 0;
}

