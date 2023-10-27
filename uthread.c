#include "uthread.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_THREADS 100

// 定义线程控制块队列
struct uthread_queue {
    struct uthread *queue[MAX_THREADS];
    int front, rear;
};

static struct uthread *current_thread = NULL;
static struct uthread *main_thread = NULL;
static struct uthread_queue ready_queue;

// 初始化队列
void init_queue(struct uthread_queue *q) {
    q->front = q->rear = -1;
}

// 将线程插入队列尾部
void enqueue(struct uthread_queue *q, struct uthread *thread) {
    if (q->rear == MAX_THREADS - 1) {
        printf("Queue is full\n");
        exit(-1);
    }
    q->queue[++q->rear] = thread;
    if (q->front == -1) {
        q->front = 0;
    }
}

// 从队列头部移除线程
struct uthread *dequeue(struct uthread_queue *q) {
    if (q->front == -1) {
        printf("Queue is empty\n");
        exit(-1);
    }
    struct uthread *thread = q->queue[q->front];
    // 将后面的元素向前移动一位
    for (int i = q->front; i < q->rear; i++) {
        q->queue[i] = q->queue[i + 1];
    }

    // 更新队列的状态
    if (q->front == q->rear) {
        q->front = q->rear = -1;
    } else {
        q->rear--;
    }
    return thread;
}

/// @brief 切换上下文
/// @param from 当前上下文
/// @param to 要切换到的上下文
extern void thread_switch(struct context *from, struct context *to);

/// @brief 线程的入口函数
/// @param tcb 线程的控制块
/// @param thread_func 线程的执行函数
/// @param arg 线程的参数
void _uthread_entry(struct uthread *tcb, void (*thread_func)(void *),
                    void *arg);

/// @brief 清空上下文结构体
/// @param context 上下文结构体指针
static inline void make_dummpy_context(struct context *context) {
  memset((struct context *)context, 0, sizeof(struct context));
}

struct uthread *uthread_create(void (*func)(void *), void *arg,const char* thread_name) {
  struct uthread *uthread = malloc(sizeof(struct uthread));

  // 申请一块16字节对齐的内存
  

  //         +------------------------+
  // low     |                        |
  //         |                        |
  //         |                        |
  //         |         stack          |
  //         |                        |
  //         |                        |
  //         |                        |
  //         +------------------------+
  //  high   |    fake return addr    |
  //         +------------------------+

  /*
  TODO: 在这里初始化uthread结构体。可能包括设置rip,rsp等寄存器。入口地址需要是函数_uthread_entry.
        除此以外，还需要设置uthread上的一些状态，保存参数等等。
        
        你需要注意rsp寄存器在这里要8字节对齐，否则后面从context switch进入其他函数的时候会有rsp寄存器
        不对齐的情况（表现为在printf里面Segment Fault）
  */
  make_dummpy_context(&uthread->context);

  long long sp;
  sp = ((long long)&uthread->stack + STACK_SIZE) & (~(long long)15);
  sp -= 8;
  uthread->context.rsp = sp; 
  uthread->context.rip = (long long)_uthread_entry;
  uthread->context.rdi = (long long)uthread;
  uthread->context.rsi = (long long)func;
  uthread->context.rdx = (long long)arg;
  uthread->name = thread_name;
  uthread->state = THREAD_INIT;
  enqueue(&ready_queue, uthread);
  return uthread;
}


void schedule() {
  /*
  TODO: 在这里写调度子线程的机制。这里需要实现一个FIFO队列。这意味着你需要一个额外的队列来保存目前活跃
        的线程。一个基本的思路是，从队列中取出线程，然后使用resume恢复函数上下文。重复这一过程。
  */
  // 如果current_thread已经结束,释放内存
  if (current_thread->state == THREAD_STOP) {
    thread_destory(current_thread);    
  }
  // 如果列为空则切换回main并且释放空间
  if (ready_queue.front == -1) {
    struct uthread *previous_thread = current_thread;
    current_thread = main_thread;
    thread_switch(&previous_thread->context, &current_thread->context);
    current_thread->state = THREAD_STOP;
    thread_destory(current_thread);
  }
  // 从队列中取出下一个线程并切换到其上下文
  struct uthread *next_thread = dequeue(&ready_queue);
  if (next_thread->state == THREAD_INIT) {
    struct uthread *previous_thread = current_thread;
    current_thread = next_thread;
    thread_switch(&previous_thread->context, &current_thread->context);
  } else if (next_thread->state == THREAD_SUSPENDED) {
    uthread_resume(next_thread);
  } else if (next_thread->state == THREAD_STOP) {
    printf("worry");
  }

}

long long uthread_yield() {
  /*
  TODO: 用户态线程让出控制权到调度器。由正在执行的用户态函数来调用。记得调整tcb状态。
  */
  current_thread->state = THREAD_SUSPENDED;
  enqueue(&ready_queue, current_thread);
  schedule();
  return 0;
}


void uthread_resume(struct uthread *tcb) {
  /*
  TODO：调度器恢复到一个函数的上下文。
  */
  struct uthread *previous_thread = current_thread;
  current_thread = tcb;  
  // 恢复线程的执行，将其状态设置为运行
  tcb->state = THREAD_RUNNING;
  thread_switch(&previous_thread->context, &tcb->context);
}

void thread_destory(struct uthread *tcb) {
  free(tcb);
}

void _uthread_entry(struct uthread *tcb, void (*thread_func)(void *),
                    void *arg) {
  /*
  TODO: 这是所有用户态线程函数开始执行的入口。在这个函数中，你需要传参数给真正调用的函数，然后设置tcb的状态。
  */
  // 设置线程状态为运行 
  tcb->state = THREAD_RUNNING;
  // 调用线程函数并传递参数
  thread_func(arg);
  // 当线程函数执行完毕后，返回到调度器
  tcb->state = THREAD_STOP;
  schedule();
}

void init_uthreads() {
  main_thread = malloc(sizeof(struct uthread));
  make_dummpy_context(&main_thread->context);
  long long sp = ((long long)&main_thread->stack + STACK_SIZE) & (~(long long)15);
  sp -= 8;
  main_thread->context.rsp = sp;
  main_thread->state = THREAD_RUNNING;
  main_thread->name = "main_thread";
  current_thread = main_thread;
  init_queue(&ready_queue);
}