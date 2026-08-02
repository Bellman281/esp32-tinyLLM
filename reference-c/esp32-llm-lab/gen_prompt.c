// Generate from an arbitrary prompt (comma-separated token ids) using the exact
// ESP32-S3 inference code (llm.h). Greedy by default; optional temperature+top-k
// sampling as a 4th arg (0 = greedy, like the chip; >0 = sample).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "llm.h"
#include "vocab.h"
static uint8_t *rf(const char*p,size_t*n){FILE*f=fopen(p,"rb");if(!f){perror(p);exit(1);}fseek(f,0,SEEK_END);*n=ftell(f);fseek(f,0,SEEK_SET);uint8_t*b=malloc(*n);if(fread(b,1,*n,f)!=*n)exit(1);fclose(f);return b;}
static void emit(int t){if(t>=VOCAB_N)return;const unsigned char*b=VOCAB_BLOB+VOCAB_OFF[t];fwrite(b,1,VOCAB_OFF[t+1]-VOCAB_OFF[t],stdout);fflush(stdout);}
static int pick(float*logits,float temp){
  if(temp<=0.f){int b=0;float bv=-1e30f;for(int x=0;x<VOCAB_N;x++)if(logits[x]>bv){bv=logits[x];b=x;}return b;}
  // top-k=40 + temperature
  int K=40; if(K>VOCAB_N)K=VOCAB_N;
  int idx[40]; float val[40];
  char used[VOCAB_N==0?1:VOCAB_N]; memset(used,0,VOCAB_N);
  for(int k=0;k<K;k++){int b=-1;float bv=-1e30f;for(int x=0;x<VOCAB_N;x++)if(!used[x]&&logits[x]>bv){bv=logits[x];b=x;}used[b]=1;idx[k]=b;val[k]=bv;}
  float mx=val[0],sum=0.f; for(int k=0;k<K;k++){val[k]=expf((val[k]-mx)/temp);sum+=val[k];}
  float r=((float)rand()/(float)RAND_MAX)*sum,acc=0.f;
  for(int k=0;k<K;k++){acc+=val[k];if(r<=acc)return idx[k];}
  return idx[K-1];
}
int main(int c,char**v){
  if(c<4){fprintf(stderr,"usage: %s model.bin N id,id[,id...] [temp]\n",v[0]);return 1;}
  size_t n;uint8_t*buf=rf(v[1],&n);Model m;if(llm_load(buf,&m)){fprintf(stderr,"bad magic\n");return 1;}
  int D=m.c.dim,L=m.c.n_layers,P=m.c.ple_dim,F=m.c.ffn,V=m.c.vocab,S=m.c.seq_len,N=atoi(v[2]);
  float temp=c>4?atof(v[4]):0.f; srand((unsigned)time(NULL));
  int prompt[512],np=0;char*t=strtok(v[3],",");while(t&&np<512){prompt[np++]=atoi(t);t=strtok(NULL,",");}
  Scratch s;s.x=malloc(D*4);s.h=malloc((F>D?F:D)*4);s.qkv=malloc(3*D*4);s.att=malloc(D*4);
  s.g1=malloc(F*4);s.g2=malloc((P>F?P:F)*4);s.ple=malloc(L*P*4);s.tmpP=malloc(L*P*4);s.trow=malloc(L*P*4);
  s.logits=malloc(V*4);s.scores=malloc(S*4);s.kcache=malloc((size_t)L*S*D*4);s.vcache=malloc((size_t)L*S*D*4);
  m.tok_emb.rows=VOCAB_N;int pos=0,tk=0;
  printf(">>> ");for(int i=0;i<np&&pos<S;i++){tk=prompt[i];emit(tk);llm_forward(&m,tk,pos++,&s);}
  for(int st=0;st<N&&pos<S;st++){tk=pick(s.logits,temp);emit(tk);llm_forward(&m,tk,pos++,&s);}
  printf("\n");return 0;
}
