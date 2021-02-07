

GenerateNodeSourceValues <- function(htree) {
  ## Generate nodes 
  frame <- htree$frame
  isLeave <- frame$var == "<leaf>"
  nodes <- rep(NA, length(isLeave))
  ylevel <- attr(htree, "ylevels")
  nodes[isLeave] <- ylevel[frame$yval][isLeave]
  nodes[!isLeave] <- labels(htree)[-1][!isLeave[-length(isLeave)]]
  
  ## Source
  node <- as.numeric(row.names(frame))
  depth <- rpart:::tree.depth(node)
  source <- depth[-1] - 1
  reps <- rle(source)
  tobeAdded <- reps$values[sapply(reps$values, function(val) sum(val >= which(reps$lengths > 1))) > 0]
  update <- source %in% tobeAdded
  source[update] <- source[update] + sapply(tobeAdded, function(tobeAdd) rep(sum(which(reps$lengths > 1) <= tobeAdd), 2))
  
  treeFrame=htree$frame
  treeRules=rpart.utils::rpart.rules(fit)
  
  targetPaths=sapply(as.numeric(row.names(treeFrame)),function(x)  
    strsplit(unlist(treeRules[x]),split=","))
  
  lastStop=  sapply(1:length(targetPaths),function(x) targetPaths[[x]] 
                    [length(targetPaths[[x]])])
  
  oneBefore=  sapply(1:length(targetPaths),function(x) targetPaths[[x]] 
                     [length(targetPaths[[x]])-1])
  
  
  target=c()
  source=c()
  values=treeFrame$n
  for(i in 2:length(oneBefore))
  {
    tmpNode=oneBefore[[i]]
    q=which(lastStop==tmpNode)
    
    q=ifelse(length(q)==0,1,q)
    source=c(source,q)
    target=c(target,i)
    
  }
  source=source-1
  target=target-1
  
  ## Return 
  return(list(nodes, source, target))
  
}
