import math
class Solution:
    
    def comp_a_bigger_b(self, a, b):
        for k in b.keys():
            if k not in a:
                return False
            elif a[k] < b[k]:
                return False
        return True

    def minWindow(self, s: str, t: str) -> str:
        maps_t = {}
        for i in t:
            if i not in maps_t:
                maps_t[i] = 1
            else:
                maps_t[i] += 1
        # print(maps_t)
        stack = []
        maps_w = {}
        r = math.inf
        result = ""
        for i in range(len(s)):
            if s[i] in t:
                if s[i] not in maps_w:
                    maps_w[s[i]] = 1
                else:
                    maps_w[s[i]] += 1
                stack.append([s[i], i])
            while self.comp_a_bigger_b(maps_w, maps_t):
                check_word = stack[0][0]
                if maps_w[check_word] > maps_t[check_word]:
                    stack.pop(0)
                    maps_w[check_word] -= 1
                else:
                    break
            if self.comp_a_bigger_b(maps_w, maps_t):
                if r > (i - stack[0][1] + 1):
                    r = (i - stack[0][1] + 1)
                    result = s[stack[0][1]: i+1]
        return result