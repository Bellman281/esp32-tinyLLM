// Run the EXACT ESP32-S3 inference (firmware/common/llm.h) on your laptop and
// print a story. No PyTorch, no emulator, no hardware -- just a C compiler.
//   cc -O3 -o gen host_generate.c -lm
//   ./gen model.bin 120
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "llm.h"
#include "vocab.h"

static uint8_t *read_file(const char *p, size_t *n){
  FILE *f=fopen(p,"rb"); if(!f){perror(p);exit(1);}
  fseek(f,0,SEEK_END); *n=ftell(f); fseek(f,0,SEEK_SET);
  uint8_t *b=malloc(*n); if(fread(b,1,*n,f)!=*n){fprintf(stderr,"short read\n");exit(1);}
  fclose(f); return b;
}
static void emit(int tok){
  if(tok>=VOCAB_N) return;
  const unsigned char *b=VOCAB_BLOB+VOCAB_OFF[tok];
  int len=VOCAB_OFF[tok+1]-VOCAB_OFF[tok];
  fwrite(b,1,len,stdout); fflush(stdout);
}
int main(int argc,char**argv){
  const char *bin = argc>1?argv[1]:"model.bin";
  int N = argc>2?atoi(argv[2]):120;
  size_t n; uint8_t *buf=read_file(bin,&n);
  Model m; if(llm_load(buf,&m)){fprintf(stderr,"bad model magic\n");return 1;}
  int D=m.c.dim,L=m.c.n_layers,P=m.c.ple_dim,F=m.c.ffn,V=m.c.vocab,S=m.c.seq_len;
  fprintf(stderr,"model: V=%d D=%d L=%d H=%d F=%d P=%d  (%.2f MB)\n",
          V,D,L,m.c.n_heads,F,P,n/1e6);
  Scratch s;
  s.x=malloc(D*4); s.h=malloc((F>D?F:D)*4); s.qkv=malloc(3*D*4); s.att=malloc(D*4);
  s.g1=malloc(F*4); s.g2=malloc((P>F?P:F)*4); s.ple=malloc(L*P*4);
  s.tmpP=malloc(L*P*4); s.trow=malloc(L*P*4); s.logits=malloc(V*4); s.scores=malloc(S*4);
  s.kcache=malloc((size_t)L*S*D*4); s.vcache=malloc((size_t)L*S*D*4);
  m.tok_emb.rows = VOCAB_N;                 // skip padded (never-emitted) rows
  int PROMPT[]={336,337,258,338};           // "Once upon a time" (this tokenizer)
  int np=4, pos=0, tok=0;
  printf(">>> ");
  for(int i=0;i<np;i++){ tok=PROMPT[i]; emit(tok); llm_forward(&m,tok,pos++,&s); }
  for(int step=0; step<N && pos<S; step++){
    int best=0; float bv=-1e30f;
    for(int v=0; v<VOCAB_N; v++) if(s.logits[v]>bv){bv=s.logits[v]; best=v;}
    tok=best; emit(tok); llm_forward(&m,tok,pos++,&s);
  }
  printf("\n");
  return 0;
}
