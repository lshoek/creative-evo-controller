for /r "images_observations/" %%i in (*.jpg) do ffmpeg -y -i %%i -vf negate %%~pi%%~ni.png