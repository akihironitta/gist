clean:
	rm -rf $(find . -name "mlruns")
	rm -rf $(find . -name "lightning_log")
	rm -rf $(find . -name "lightning_logs")
	rm -rf _ckpt_*
	rm -rf .mypy_cache
	rm -rf .pytest_cache
