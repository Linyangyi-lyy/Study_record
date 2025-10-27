# Git 学习

## <font color=" grey"> 一、常用命令  </font> 

* `git commit` ：推送当前版本

  

* `git branch <name>` ：创建当前版本新分支

* `git checkout <name>`：切换到当前分支

* ` git checkout -b <name> `：创建并切换到新的分支

  

* `git merge <name>`：将name分支合并到当前分支a（a*, name  =>  a *）

* `git rebase <name>`：将当前分支a与name分支合并，成为name分支的线性关系（name  ->  a）

* `git rebase <name1> <name2>`= `git checkout <newbase> + git rebase <main>`

  : 先切换到name2分支，再将其更新到name1分支 (name2, name1 -> name2 ' )



* `git fetch`：将仓库数据拉回下载（本地仓库缺失的分支）到本地仓库（=下载操作，并未改变本地文件）
* `git pull`：获取远程仓库数据，并与当前分支merge合并 （=`git fetch + git merge`）
* `git push`：将本地分支推送到远程仓库
* ` git pull --rebase `：获取远程仓库数据，并与当前分支rebase合并（= `git fetch + fit rebase`）



## <font color=" grey"> 二、创建仓库-推送-拉取 （实操代码记录）</font> 

```git
# 初始化代码文件
git init
git add .
git commit -m 'initial project version'

# 在github创建仓库（如果代码里没有README，创建仓库时就不要增加）

# 将代码推送到仓库
git remote add origin https://github.com/Linyangyi-lyy/Test.git（github的URL）
git push -u origin master（下次在同个分支上操作就只用 git push 即可）

# 更改仓库代码后拉取到本地
git pull

# 在本地新增文件并更改后，再次推送到远端仓库
git add . 
git commit -m "increase learning records"
git push

#********************************** 加入分支的运用 ****************************************#

```











