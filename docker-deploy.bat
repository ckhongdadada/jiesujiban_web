@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

echo ==============================================
echo   接诉即办智能服务系统 - Docker 部署
echo ==============================================
echo.

REM 检查 Docker
where docker >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [错误] Docker 未安装
    echo 请访问 https://docs.docker.com/get-docker/ 安装 Docker
    exit /b 1
)

docker compose version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [错误] Docker Compose 插件未安装
    echo 请访问 https://docs.docker.com/compose/install/ 安装 Docker Compose
    exit /b 1
)

echo [OK] Docker 环境检查通过

REM 检查环境文件
if not exist ".env" (
    echo [提示] 未找到 .env 文件，使用默认配置
    copy docker\.env.example .env >nul
)

REM 检查目录
if not exist "checkpoints" mkdir checkpoints
if not exist "data\runtime" mkdir data\runtime
if not exist "logs" mkdir logs

REM 解析命令
set COMMAND=%1
if "%COMMAND%"=="" set COMMAND=up

if "%COMMAND%"=="up" goto :up
if "%COMMAND%"=="down" goto :down
if "%COMMAND%"=="restart" goto :restart
if "%COMMAND%"=="logs" goto :logs
if "%COMMAND%"=="build" goto :build
if "%COMMAND%"=="ps" goto :ps
goto :usage

:up
echo.
echo 启动服务...
docker compose -p jsjb up -d
echo.
echo ==============================================
echo   服务启动成功！
echo ==============================================
echo.
echo 访问地址:
echo   - API 服务:    http://localhost:5000
echo   - Neo4j 控制台: http://localhost:7474
echo.
echo 查看日志: docker compose -p jsjb logs -f app
echo 停止服务: docker-deploy.bat down
goto :end

:down
echo 停止服务...
docker compose -p jsjb down
echo [OK] 服务已停止
goto :end

:restart
echo 重启服务...
docker compose -p jsjb restart
echo [OK] 服务已重启
goto :end

:logs
docker compose -p jsjb logs -f %2
goto :end

:build
echo 构建镜像...
docker compose -p jsjb build --no-cache
echo [OK] 镜像构建完成
goto :end

:ps
docker compose -p jsjb ps
goto :end

:usage
echo 用法: %0 {up^|down^|restart^|logs^|build^|ps}
echo.
echo 命令说明:
echo   up      - 启动所有服务
echo   down    - 停止所有服务
echo   restart - 重启所有服务
echo   logs    - 查看日志 (可指定服务名)
echo   build   - 重新构建镜像
echo   ps      - 查看服务状态
exit /b 1

:end
endlocal
