/**
 * WebLLM engine worker — keeps GPU compile + inference off the UI thread.
 * Official handler from @mlc-ai/web-llm.
 */
import { WebWorkerMLCEngineHandler } from 'https://esm.run/@mlc-ai/web-llm';

const handler = new WebWorkerMLCEngineHandler();
self.onmessage = (msg) => handler.onmessage(msg);
